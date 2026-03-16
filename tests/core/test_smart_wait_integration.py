from unittest.mock import Mock, patch

import pytest

from agentm.core.task_manager import TaskManager
from agentm.models.data import ManagedTask
from agentm.models.enums import AgentRunStatus

pytestmark = pytest.mark.asyncio


class TestSmartWaitIntegration:
    """Test smart wait strategy integration with check_tasks."""

    @pytest.fixture
    def task_manager(self):
        """Create a TaskManager instance."""
        return TaskManager()

    @pytest.fixture
    def mock_agent_pool(self):
        """Create a mock agent pool."""
        pool = Mock()
        pool.worker_max_steps = 10
        pool.worker_self_reports_trajectory = False
        pool.create_worker.return_value = Mock()
        return pool

    async def test_check_tasks_uses_smart_wait_when_none_specified(
        self, task_manager, mock_agent_pool
    ):
        """Test that check_tasks uses smart wait when wait_seconds is None."""
        from agentm.tools.orchestrator import create_orchestrator_tools

        # Create orchestrator tools
        tools = create_orchestrator_tools(task_manager, mock_agent_pool)
        check_tasks = tools["check_tasks"]

        # Submit a task
        task_id = await task_manager.submit(
            "test-agent", "test instruction", "scout", max_steps=10
        )

        # Mock the smart wait strategy
        with patch.object(
            task_manager._smart_wait_strategy,
            "calculate_wait_time",
            return_value=25.0,
        ) as mock_calc:
            # Call check_tasks without specifying wait_seconds
            result = await check_tasks("waiting for task")

            # Verify smart wait was used
            mock_calc.assert_called_once()
            call_args = mock_calc.call_args[0]
            assert call_args[0] == task_id  # task_id
            assert call_args[1] == "test-agent"  # agent_id
            assert call_args[2] == "scout"  # task_type
            assert call_args[3] >= 0  # elapsed_seconds
            assert call_args[4] == 0  # current_step
            assert call_args[5] == 10  # max_steps

    async def test_check_tasks_records_completion_time(
        self, task_manager, mock_agent_pool
    ):
        """Test that check_tasks records completion time for finished tasks."""
        from agentm.tools.orchestrator import create_orchestrator_tools

        # Create orchestrator tools
        tools = create_orchestrator_tools(task_manager, mock_agent_pool)
        check_tasks = tools["check_tasks"]

        # Create a completed task
        task = ManagedTask(
            task_id="test-task",
            agent_id="test-agent",
            instruction="test",
            status=AgentRunStatus.COMPLETED,
            started_at="2024-01-01T10:00:00",
            completed_at="2024-01-01T10:02:30",  # 2.5 minutes
            duration_seconds=150.0,
            current_step=5,
            max_steps=5,
        )
        task.reported = False  # Not yet reported
        task_manager._tasks["test-task"] = task

        # Mock the completion recording
        with patch.object(
            task_manager._smart_wait_strategy, "record_completion"
        ) as mock_record:
            # Call check_tasks
            result = await check_tasks("collecting results")

            # Verify completion was recorded
            mock_record.assert_called_once_with("test-task", 150.0)

            # Verify task is marked as reported
            assert task.reported is True

    async def test_smart_wait_adapts_to_task_progress(self, task_manager):
        """Test that smart wait time adapts based on task progress."""
        strategy = task_manager._smart_wait_strategy

        # Test early stage (0-60s)
        wait_time = strategy.calculate_wait_time("task1", "agent1", "scout", 30, 2, 10)
        assert 15 <= wait_time <= 20  # Base wait + small increase

        # Test near completion (90%+ progress)
        wait_time = strategy.calculate_wait_time("task1", "agent1", "scout", 120, 9, 10)
        assert wait_time == 6  # Fixed short wait when almost done

        # Test most way done (70%+ progress)
        wait_time = strategy.calculate_wait_time("task1", "agent1", "scout", 90, 7, 10)
        assert wait_time == int(15 * 0.7)  # 70% of base wait

    async def test_smart_wait_exponential_backoff(self, task_manager):
        """Test exponential backoff for long-running tasks."""
        strategy = task_manager._smart_wait_strategy

        # Test with progress < 70% to avoid progress-based adjustments
        task_id = "long-task"
        base_wait = 15  # scout task base wait

        # First iteration (2 minutes elapsed, 0 previous waits, 50% progress)
        wait1 = strategy.calculate_wait_time(task_id, "agent1", "scout", 120, 5, 10)
        # For 2-3 minutes: factor = 1.3^0 = 1, so wait = base_wait = 15
        expected1 = min(int(base_wait * (1.3**0)), 45)
        assert wait1 == expected1

        # Second iteration (3 minutes elapsed, 1 previous wait, 60% progress)
        wait2 = strategy.calculate_wait_time(task_id, "agent1", "scout", 180, 6, 10)
        # For 2-3 minutes with 1 iteration: factor = 1.3^1 = 1.3, so wait = 15 * 1.3 = 19.5 -> 19
        expected2 = min(int(base_wait * (1.3**1)), 45)
        assert wait2 == expected2

        # Test with >3 minutes and 0 iterations (new task)
        wait3 = strategy.calculate_wait_time(
            "new-task",
            "agent1",
            "scout",
            300,
            2,
            10,  # 20% progress
        )
        # For >3 minutes with 0 iterations: factor = 1.5^0 = 1, so wait = 15 * 1 = 15
        expected3 = min(int(base_wait * (1.5**0)), 90)
        assert wait3 == expected3

    async def test_different_task_types_have_different_waits(self, task_manager):
        """Test that different task types have different base wait times."""
        strategy = task_manager._smart_wait_strategy

        # Scout tasks
        scout_wait = strategy.calculate_wait_time(
            "scout-task", "agent1", "scout", 30, 0, 10
        )
        assert scout_wait == 15  # Base wait for scout

        # Verify tasks
        verify_wait = strategy.calculate_wait_time(
            "verify-task", "agent1", "verify", 30, 0, 10
        )
        assert verify_wait == 10  # Base wait for verify

        # Deep analyze tasks
        analyze_wait = strategy.calculate_wait_time(
            "analyze-task", "agent1", "deep_analyze", 30, 0, 10
        )
        assert analyze_wait == 20  # Base wait for deep_analyze

    async def test_wait_history_isolation_between_tasks(self, task_manager):
        """Test that wait history is properly isolated between different tasks."""
        strategy = task_manager._smart_wait_strategy

        # Task 1 has multiple waits
        for i in range(3):
            wait = strategy.calculate_wait_time("task1", "agent1", "scout", 60, 3, 10)

        # Task 2 should start fresh (no history)
        wait = strategy.calculate_wait_time("task2", "agent1", "scout", 60, 3, 10)
        # Should be base wait + 0 (no iterations)
        assert wait == 15

    async def test_memory_cleanup_on_completion(self, task_manager):
        """Test that task history is cleaned up after completion."""
        strategy = task_manager._smart_wait_strategy

        # Build some history
        task_id = "complete-task"
        for i in range(3):
            strategy.calculate_wait_time(task_id, "agent1", "scout", 60 + i * 10, 3, 10)

        # Verify history exists
        assert task_id in strategy._wait_history
        assert len(strategy._wait_history[task_id]) == 3

        # Record completion
        strategy.record_completion(task_id, 180.0)

        # Verify history is cleaned up
        assert task_id not in strategy._wait_history

    async def test_check_tasks_with_no_running_tasks(
        self, task_manager, mock_agent_pool
    ):
        """Test check_tasks behavior when no tasks are running."""
        from agentm.tools.orchestrator import create_orchestrator_tools

        # Create orchestrator tools
        tools = create_orchestrator_tools(task_manager, mock_agent_pool)
        check_tasks = tools["check_tasks"]

        # Call check_tasks with no running tasks
        result = await check_tasks("no tasks running")

        # Should return immediately with wait_seconds=0
        assert result is not None

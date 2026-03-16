from agentm.core.task_manager import SmartWaitStrategy


class TestSmartWaitStrategy:
    """Test the SmartWaitStrategy class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = SmartWaitStrategy()

    def test_initial_wait_times_by_task_type(self):
        """Test that different task types get appropriate initial wait times."""
        # Scout tasks should start with 15s
        wait_time = self.strategy.calculate_wait_time(
            task_id="test-1",
            agent_id="scout-1",
            task_type="scout",
            elapsed_seconds=5,
            current_step=1,
            max_steps=10,
        )
        assert wait_time == 15

        # Verify tasks should start with 10s
        wait_time = self.strategy.calculate_wait_time(
            task_id="test-2",
            agent_id="verify-1",
            task_type="verify",
            elapsed_seconds=5,
            current_step=1,
            max_steps=10,
        )
        assert wait_time == 10

        # Deep analyze tasks should start with 20s
        wait_time = self.strategy.calculate_wait_time(
            task_id="test-3",
            agent_id="deep-1",
            task_type="deep_analyze",
            elapsed_seconds=5,
            current_step=1,
            max_steps=10,
        )
        assert wait_time == 20

        # Unknown task types default to 12s
        wait_time = self.strategy.calculate_wait_time(
            task_id="test-4",
            agent_id="unknown-1",
            task_type="unknown",
            elapsed_seconds=5,
            current_step=1,
            max_steps=10,
        )
        assert wait_time == 12

    def test_progress_based_adjustments(self):
        """Test that wait time adjusts based on task progress."""
        # Near completion (>90% progress) should reduce wait time
        wait_time = self.strategy.calculate_wait_time(
            task_id="test-1",
            agent_id="scout-1",
            task_type="scout",
            elapsed_seconds=100,
            current_step=9,
            max_steps=10,
        )
        assert wait_time == 6  # Fixed 6s for >90% progress

        # Most way done (>70% progress) should use moderate wait
        wait_time = self.strategy.calculate_wait_time(
            task_id="test-2",
            agent_id="scout-1",
            task_type="scout",
            elapsed_seconds=100,
            current_step=7,
            max_steps=10,
        )
        assert wait_time == 10  # 15 * 0.7 = 10.5, rounded to 10

    def test_time_based_exponential_backoff(self):
        """Test exponential backoff based on elapsed time."""
        # After 3 minutes, should use exponential backoff
        wait_time = self.strategy.calculate_wait_time(
            task_id="test-1",
            agent_id="scout-1",
            task_type="scout",
            elapsed_seconds=200,  # > 3 minutes
            current_step=5,
            max_steps=10,
        )
        # First iteration: 15 * 1.5^0 = 15 (base value)
        assert wait_time == 15

        # Second iteration should increase
        wait_time2 = self.strategy.calculate_wait_time(
            task_id="test-1",  # Same task ID
            agent_id="scout-1",
            task_type="scout",
            elapsed_seconds=200,
            current_step=5,
            max_steps=10,
        )
        # Second iteration: 15 * 1.5^1 = 22.5, capped at 90
        assert wait_time2 == 22

    def test_wait_time_capping(self):
        """Test that wait times are properly capped."""
        # Should not exceed 90 seconds even with many iterations
        for i in range(10):
            wait_time = self.strategy.calculate_wait_time(
                task_id="test-long",
                agent_id="scout-1",
                task_type="scout",
                elapsed_seconds=300,  # > 5 minutes
                current_step=5,
                max_steps=10,
            )
        assert wait_time <= 90

    def test_early_stage_progressive_increase(self):
        """Test progressive increase in early stages."""
        # In early stage (< 1 minute), wait time increases by 2s per iteration
        wait_time1 = self.strategy.calculate_wait_time(
            task_id="test-early",
            agent_id="scout-1",
            task_type="scout",
            elapsed_seconds=30,
            current_step=2,
            max_steps=10,
        )
        assert wait_time1 == 15  # Base wait

        wait_time2 = self.strategy.calculate_wait_time(
            task_id="test-early",  # Same task
            agent_id="scout-1",
            task_type="scout",
            elapsed_seconds=30,
            current_step=2,
            max_steps=10,
        )
        assert wait_time2 == 17  # 15 + 2

        wait_time3 = self.strategy.calculate_wait_time(
            task_id="test-early",  # Same task
            agent_id="scout-1",
            task_type="scout",
            elapsed_seconds=30,
            current_step=2,
            max_steps=10,
        )
        assert wait_time3 == 19  # 15 + 4

    def test_completion_recording(self):
        """Test that completion times are properly recorded."""
        # Add some wait history
        self.strategy.calculate_wait_time(
            task_id="test-complete",
            agent_id="scout-1",
            task_type="scout",
            elapsed_seconds=100,
            current_step=5,
            max_steps=10,
        )

        # Record completion
        self.strategy.record_completion("test-complete", 150.0)

        # History should be cleaned up
        assert "test-complete" not in self.strategy._wait_history

    def test_mixed_scenarios(self):
        """Test realistic mixed scenarios."""
        # Scenario 1: Scout task running for 4 minutes, 60% complete
        wait_time = self.strategy.calculate_wait_time(
            task_id="scout-real",
            agent_id="scout-1",
            task_type="scout",
            elapsed_seconds=240,  # 4 minutes
            current_step=6,
            max_steps=10,
        )
        # Should use time-based backoff but not too aggressive due to progress
        assert 15 <= wait_time <= 45

        # Scenario 2: Verify task near completion
        wait_time = self.strategy.calculate_wait_time(
            task_id="verify-near-done",
            agent_id="verify-1",
            task_type="verify",
            elapsed_seconds=90,
            current_step=9,
            max_steps=10,
        )
        # Should be very short wait (6 for verify near completion)
        assert wait_time == 6

    def test_iteration_isolation(self):
        """Test that different tasks don't interfere with each other's iterations."""
        # Task A: Multiple iterations
        for i in range(3):
            self.strategy.calculate_wait_time(
                task_id="task-a",
                agent_id="scout-1",
                task_type="scout",
                elapsed_seconds=200,
                current_step=5,
                max_steps=10,
            )

        # Task B: First iteration
        wait_time_b = self.strategy.calculate_wait_time(
            task_id="task-b",
            agent_id="scout-1",
            task_type="scout",
            elapsed_seconds=200,
            current_step=5,
            max_steps=10,
        )

        # Task B should start from iteration 0, not be affected by Task A
        assert wait_time_b == 15  # First iteration uses base wait time

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Zero elapsed time
        wait_time = self.strategy.calculate_wait_time(
            task_id="test-zero",
            agent_id="scout-1",
            task_type="scout",
            elapsed_seconds=0,
            current_step=0,
            max_steps=10,
        )
        assert wait_time == 15  # Base wait

        # Zero max steps (shouldn't crash)
        wait_time = self.strategy.calculate_wait_time(
            task_id="test-no-max",
            agent_id="scout-1",
            task_type="scout",
            elapsed_seconds=100,
            current_step=5,
            max_steps=0,
        )
        assert wait_time >= 15  # Should still work

        # Very high progress
        wait_time = self.strategy.calculate_wait_time(
            task_id="test-almost-done",
            agent_id="scout-1",
            task_type="scout",
            elapsed_seconds=100,
            current_step=99,
            max_steps=100,
        )
        assert wait_time == 6  # Minimum wait for >90% progress

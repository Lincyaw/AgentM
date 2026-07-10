# Rescue Window report

- prefixes: 147 | trajectories: 30 | cases: 30
- epsilon: 0.0
- oracle opportunity (all): 95.2%
- oracle opportunity (failing baseline): 95.2%
- mean G*: 0.359 | mean gap: 0.182
- rescue rate: 0.2% (n=3033) | harm rate: - (n=0)
- harm-sensitive prefixes: 52.4%
- window: exists 100.0% | mean width 0.583 | mean area 0.217 | mean peak 0.449

## Per-prefix

| prefix | progress | cont✓ | G* | best | gap | labels |
|---|---|---|---|---|---|---|
| 0f4317a663494b099a3e0e6fba3fef9f#t10 | 0.5556 | False | 0.597 | ADVISE:ORACLE_DIAG | 0.075 | high_continuation_risk, oracle_actionable |
| 0f4317a663494b099a3e0e6fba3fef9f#t14 | 0.7778 | False | 0.557 | ADVISE:ORACLE_DIAG | 0.088 | high_continuation_risk, oracle_actionable |
| 0f4317a663494b099a3e0e6fba3fef9f#t16 | 0.8889 | False | 0.733 | ADVISE:ORACLE_DIAG | 0.227 | high_continuation_risk, oracle_actionable |
| 0f4317a663494b099a3e0e6fba3fef9f#t3 | 0.1667 | False | 0.536 | ADVISE:ORACLE_DIAG | 0.010 | high_continuation_risk, oracle_actionable |
| 0f4317a663494b099a3e0e6fba3fef9f#t7 | 0.3889 | False | 0.600 | ADVISE:ORACLE_DIAG | 0.053 | high_continuation_risk, oracle_actionable |
| 1364aff6640c476f9376a3775c8f47b3#t10 | 0.8333 | False | 0.458 | ADVISE:ORACLE_DIAG | 0.029 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 1364aff6640c476f9376a3775c8f47b3#t3 | 0.25 | False | 0.545 | ADVISE:ORACLE_DIAG | 0.393 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 1364aff6640c476f9376a3775c8f47b3#t4 | 0.3333 | False | 0.470 | ADVISE:ORACLE_DIAG | 0.279 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 1364aff6640c476f9376a3775c8f47b3#t7 | 0.5833 | False | 0.468 | ADVISE:ORACLE_DIAG | 0.239 | high_continuation_risk, oracle_actionable |
| 1364aff6640c476f9376a3775c8f47b3#t9 | 0.75 | False | 0.496 | ADVISE:ORACLE_DIAG | 0.169 | high_continuation_risk, oracle_actionable |
| 2123e72d14ae47c09ecc4c67e74c5d2f#t10 | 0.5882 | False | 0.500 | ADVISE:ORACLE_DIAG | 0.095 | high_continuation_risk, oracle_actionable |
| 2123e72d14ae47c09ecc4c67e74c5d2f#t13 | 0.7647 | False | 0.442 | ADVISE:ORACLE_DIAG | 0.143 | high_continuation_risk, oracle_actionable |
| 2123e72d14ae47c09ecc4c67e74c5d2f#t15 | 0.8824 | False | 0.310 | ADVISE:ORACLE_DIAG | 0.185 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 2123e72d14ae47c09ecc4c67e74c5d2f#t3 | 0.1765 | False | 0.433 | ADVISE:ORACLE_DIAG | 0.060 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 2123e72d14ae47c09ecc4c67e74c5d2f#t6 | 0.3529 | False | 0.542 | ADVISE:ORACLE_DIAG | 0.163 | high_continuation_risk, oracle_actionable |
| 343d923ede0e4d3abb4eb827b36d4e7a#t11 | 0.7333 | False | 0.512 | ADVISE:ORACLE_DIAG | 0.345 | high_continuation_risk, oracle_actionable |
| 343d923ede0e4d3abb4eb827b36d4e7a#t13 | 0.8667 | False | 0.470 | ADVISE:ORACLE_DIAG | 0.430 | high_continuation_risk, oracle_actionable |
| 343d923ede0e4d3abb4eb827b36d4e7a#t3 | 0.2 | False | 0.600 | ADVISE:ORACLE_DIAG | 0.477 | high_continuation_risk, oracle_actionable |
| 343d923ede0e4d3abb4eb827b36d4e7a#t6 | 0.4 | False | 0.600 | ADVISE:ORACLE_DIAG | 0.375 | high_continuation_risk, oracle_actionable |
| 343d923ede0e4d3abb4eb827b36d4e7a#t8 | 0.5333 | False | 0.339 | ADVISE:ORACLE_DIAG | 0.187 | high_continuation_risk, oracle_actionable |
| 348492fed1ea47f08eca5e6ec3380933#t3 | 0.2727 | False | 0.583 | ADVISE:ORACLE_DIAG | 0.120 | high_continuation_risk, oracle_actionable |
| 348492fed1ea47f08eca5e6ec3380933#t4 | 0.3636 | False | 0.500 | ADVISE:ORACLE_DIAG | 0.461 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 348492fed1ea47f08eca5e6ec3380933#t6 | 0.5455 | None | 0.583 | ADVISE:ORACLE_DIAG | 0.315 | oracle_actionable |
| 348492fed1ea47f08eca5e6ec3380933#t8 | 0.7273 | None | 0.542 | ADVISE:ORACLE_DIAG | 0.510 | oracle_actionable, harm_sensitive |
| 348492fed1ea47f08eca5e6ec3380933#t9 | 0.8182 | False | 0.597 | ADVISE:ORACLE_DIAG | 0.523 | high_continuation_risk, oracle_actionable |
| 360fabaf623845fb93a010397b49cae0#t3 | 0.3333 | False | 0.578 | REPLAN:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable |
| 360fabaf623845fb93a010397b49cae0#t5 | 0.5556 | False | 0.447 | REPLAN:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable |
| 360fabaf623845fb93a010397b49cae0#t6 | 0.6667 | False | 0.452 | ADVISE:ORACLE_DIAG | 0.024 | high_continuation_risk, oracle_actionable |
| 360fabaf623845fb93a010397b49cae0#t7 | 0.7778 | False | 0.302 | ADVISE:ORACLE_DIAG | 0.066 | high_continuation_risk, oracle_actionable |
| 3bf7a35738d04bcb88d2842af6a840a6#t3 | 0.3333 | False | 0.236 | ADVISE:ORACLE_DIAG | 0.068 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 3bf7a35738d04bcb88d2842af6a840a6#t5 | 0.5556 | False | 0.389 | ADVISE:ORACLE_DIAG | 0.090 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 3bf7a35738d04bcb88d2842af6a840a6#t6 | 0.6667 | False | 0.479 | ADVISE:ORACLE_DIAG | 0.153 | high_continuation_risk, oracle_actionable |
| 3bf7a35738d04bcb88d2842af6a840a6#t7 | 0.7778 | False | 0.446 | ADVISE:ORACLE_DIAG | 0.206 | high_continuation_risk, oracle_actionable |
| 3d1ac499720d447ea44aa266fd65ab45#t10 | 0.7143 | False | 0.289 | GENERIC:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 3d1ac499720d447ea44aa266fd65ab45#t12 | 0.8571 | False | 0.271 | VERIFY:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable |
| 3d1ac499720d447ea44aa266fd65ab45#t3 | 0.2143 | False | 0.109 | REPLAN:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 3d1ac499720d447ea44aa266fd65ab45#t5 | 0.3571 | False | -0.040 | REPLAN:TYPE_TARGET | 0.000 | high_continuation_risk, irrecoverable_under_scope, harm_sensitive |
| 3d1ac499720d447ea44aa266fd65ab45#t8 | 0.5714 | False | -0.040 | REPLAN:TYPE_TARGET | 0.000 | high_continuation_risk, irrecoverable_under_scope, harm_sensitive |
| 3dab81cc65934da08d00be2d12fb8a87#t3 | 0.2727 | False | 0.164 | GENERIC:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 3dab81cc65934da08d00be2d12fb8a87#t4 | 0.3636 | False | 0.278 | GENERIC:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 3dab81cc65934da08d00be2d12fb8a87#t6 | 0.5455 | False | 0.187 | GENERIC:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 3dab81cc65934da08d00be2d12fb8a87#t8 | 0.7273 | False | 0.142 | ADVISE:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 3dab81cc65934da08d00be2d12fb8a87#t9 | 0.8182 | False | -0.040 | ADVISE:ORACLE_DIAG | 0.092 | high_continuation_risk, irrecoverable_under_scope, harm_sensitive, channel_limited |
| 4f2c9bc7199a4b96b7cfb7521a1f1e4a#t10 | 0.7692 | False | 0.107 | ADVISE:ORACLE_DIAG | 0.024 | high_continuation_risk, oracle_actionable |
| 4f2c9bc7199a4b96b7cfb7521a1f1e4a#t11 | 0.8462 | False | 0.342 | ADVISE:ORACLE_DIAG | 0.328 | high_continuation_risk, oracle_actionable |
| 4f2c9bc7199a4b96b7cfb7521a1f1e4a#t3 | 0.2308 | False | 0.460 | ADVISE:ORACLE_DIAG | 0.428 | high_continuation_risk, oracle_actionable |
| 4f2c9bc7199a4b96b7cfb7521a1f1e4a#t5 | 0.3846 | False | 0.443 | ADVISE:ORACLE_DIAG | 0.460 | high_continuation_risk, oracle_actionable, harm_sensitive, channel_limited |
| 4f2c9bc7199a4b96b7cfb7521a1f1e4a#t7 | 0.5385 | False | 0.402 | ADVISE:ORACLE_DIAG | 0.388 | high_continuation_risk, oracle_actionable |
| 6066635caacf4998b23d54b93ef76652#t10 | 0.8333 | False | -0.051 | GENERIC:TYPE_TARGET | 0.000 | high_continuation_risk, irrecoverable_under_scope, harm_sensitive |
| 6066635caacf4998b23d54b93ef76652#t3 | 0.25 | False | 0.144 | FINAL_AUDIT:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 6066635caacf4998b23d54b93ef76652#t4 | 0.3333 | False | 0.049 | ADVISE:ORACLE_DIAG | 0.094 | high_continuation_risk, oracle_actionable, harm_sensitive, channel_limited |
| 6066635caacf4998b23d54b93ef76652#t7 | 0.5833 | False | 0.170 | ADVISE:ORACLE_DIAG | 0.014 | high_continuation_risk, oracle_actionable |
| 6066635caacf4998b23d54b93ef76652#t9 | 0.75 | False | 0.068 | VERIFY:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 6e7a99140c6844f0961577fb75bb0494#t3 | 0.3 | False | 0.103 | ADVISE:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 6e7a99140c6844f0961577fb75bb0494#t4 | 0.4 | False | 0.102 | VERIFY:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 6e7a99140c6844f0961577fb75bb0494#t5 | 0.5 | False | 0.224 | REPLAN:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable |
| 6e7a99140c6844f0961577fb75bb0494#t7 | 0.7 | False | 0.043 | FINAL_AUDIT:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 6e7a99140c6844f0961577fb75bb0494#t8 | 0.8 | False | 0.013 | VERIFY:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 751eedd6cbd2450bb9b1ed8aa4e1a1e0#t10 | 0.5882 | False | 0.467 | ADVISE:ORACLE_DIAG | 0.367 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 751eedd6cbd2450bb9b1ed8aa4e1a1e0#t13 | 0.7647 | False | 0.462 | ADVISE:ORACLE_DIAG | 0.373 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 751eedd6cbd2450bb9b1ed8aa4e1a1e0#t14 | 0.8235 | False | 0.550 | ADVISE:ORACLE_DIAG | 0.386 | high_continuation_risk, oracle_actionable |
| 751eedd6cbd2450bb9b1ed8aa4e1a1e0#t15 | 0.8824 | False | 0.292 | ADVISE:ORACLE_DIAG | 0.197 | high_continuation_risk, oracle_actionable |
| 751eedd6cbd2450bb9b1ed8aa4e1a1e0#t3 | 0.1765 | False | 0.570 | ADVISE:ORACLE_DIAG | 0.247 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 751eedd6cbd2450bb9b1ed8aa4e1a1e0#t6 | 0.3529 | False | 0.533 | ADVISE:ORACLE_DIAG | 0.196 | high_continuation_risk, oracle_actionable |
| 76c6ad21a0c84a9faffa1a16a01925bb#t10 | 0.7143 | False | 0.463 | ADVISE:ORACLE_DIAG | 0.429 | high_continuation_risk, oracle_actionable |
| 76c6ad21a0c84a9faffa1a16a01925bb#t3 | 0.2143 | False | 0.467 | ADVISE:ORACLE_DIAG | 0.359 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 76c6ad21a0c84a9faffa1a16a01925bb#t5 | 0.3571 | False | 0.432 | ADVISE:ORACLE_DIAG | 0.309 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 76c6ad21a0c84a9faffa1a16a01925bb#t8 | 0.5714 | False | 0.374 | ADVISE:ORACLE_DIAG | 0.306 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 76c6ad21a0c84a9faffa1a16a01925bb#t9 | 0.6429 | False | 0.453 | ADVISE:ORACLE_DIAG | 0.389 | high_continuation_risk, oracle_actionable |
| 77c5253679f644c39a96d22fbdbf1b70#t3 | 0.2727 | False | 0.486 | ADVISE:ORACLE_DIAG | 0.361 | high_continuation_risk, oracle_actionable |
| 77c5253679f644c39a96d22fbdbf1b70#t4 | 0.3636 | False | 0.486 | ADVISE:ORACLE_DIAG | 0.411 | high_continuation_risk, oracle_actionable |
| 77c5253679f644c39a96d22fbdbf1b70#t6 | 0.5455 | False | 0.453 | ADVISE:ORACLE_DIAG | 0.292 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 77c5253679f644c39a96d22fbdbf1b70#t8 | 0.7273 | False | 0.447 | ADVISE:ORACLE_DIAG | 0.447 | high_continuation_risk, oracle_actionable, harm_sensitive, channel_limited |
| 77c5253679f644c39a96d22fbdbf1b70#t9 | 0.8182 | False | 0.472 | ADVISE:ORACLE_DIAG | 0.447 | high_continuation_risk, oracle_actionable |
| 7e61cc7ccc21483b8837282cf6376539#t10 | 0.7143 | False | 0.022 | REPLAN:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 7e61cc7ccc21483b8837282cf6376539#t12 | 0.8571 | False | 0.067 | ADVISE:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable |
| 7e61cc7ccc21483b8837282cf6376539#t3 | 0.2143 | False | 0.133 | REPLAN:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable |
| 7e61cc7ccc21483b8837282cf6376539#t5 | 0.3571 | False | -0.013 | GENERIC:TYPE_TARGET | 0.000 | high_continuation_risk, irrecoverable_under_scope, harm_sensitive |
| 7e61cc7ccc21483b8837282cf6376539#t8 | 0.5714 | False | 0.013 | GENERIC:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 82192b014c6945279709876a3ec80113#t10 | 0.7143 | False | 0.467 | ADVISE:ORACLE_DIAG | 0.350 | high_continuation_risk, oracle_actionable |
| 82192b014c6945279709876a3ec80113#t12 | 0.8571 | False | 0.270 | ADVISE:ORACLE_DIAG | 0.270 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 82192b014c6945279709876a3ec80113#t3 | 0.2143 | False | 0.320 | ADVISE:ORACLE_DIAG | 0.231 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 82192b014c6945279709876a3ec80113#t5 | 0.3571 | False | 0.282 | ADVISE:ORACLE_DIAG | 0.333 | high_continuation_risk, oracle_actionable, harm_sensitive, channel_limited |
| 82192b014c6945279709876a3ec80113#t8 | 0.5714 | False | 0.453 | ADVISE:ORACLE_DIAG | 0.320 | high_continuation_risk, oracle_actionable |
| 830fd5625055496497a85cba62610c0c#t10 | 0.7143 | False | 0.431 | ADVISE:ORACLE_DIAG | 0.438 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 830fd5625055496497a85cba62610c0c#t12 | 0.8571 | False | 0.435 | ADVISE:ORACLE_DIAG | 0.400 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 830fd5625055496497a85cba62610c0c#t3 | 0.2143 | False | 0.451 | ADVISE:ORACLE_DIAG | 0.291 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 830fd5625055496497a85cba62610c0c#t5 | 0.3571 | False | 0.512 | ADVISE:ORACLE_DIAG | 0.323 | high_continuation_risk, oracle_actionable |
| 830fd5625055496497a85cba62610c0c#t8 | 0.5714 | False | 0.442 | ADVISE:ORACLE_DIAG | 0.238 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 88c5dae91b464c959d9d6409db59115e#t10 | 0.8333 | False | 0.086 | GENERIC:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 88c5dae91b464c959d9d6409db59115e#t3 | 0.25 | False | 0.124 | VERIFY:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 88c5dae91b464c959d9d6409db59115e#t4 | 0.3333 | False | 0.083 | REPLAN:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 88c5dae91b464c959d9d6409db59115e#t7 | 0.5833 | False | -0.060 | VERIFY:TYPE_TARGET | 0.000 | high_continuation_risk, irrecoverable_under_scope, harm_sensitive |
| 88c5dae91b464c959d9d6409db59115e#t9 | 0.75 | False | 0.087 | ADVISE:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 9504e665d0da41ec960d9d2b51b463e9#t3 | 0.2727 | False | 0.360 | ADVISE:ORACLE_DIAG | 0.030 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 9504e665d0da41ec960d9d2b51b463e9#t4 | 0.3636 | False | 0.408 | VERIFY:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 9504e665d0da41ec960d9d2b51b463e9#t6 | 0.5455 | False | 0.367 | GENERIC:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| 9504e665d0da41ec960d9d2b51b463e9#t8 | 0.7273 | False | 0.417 | ADVISE:ORACLE_DIAG | 0.097 | high_continuation_risk, oracle_actionable |
| 9504e665d0da41ec960d9d2b51b463e9#t9 | 0.8182 | False | 0.352 | ADVISE:ORACLE_DIAG | 0.009 | high_continuation_risk, oracle_actionable |
| a3ca7276563443fbb1a8291835219e58#t3 | 0.2727 | False | 0.429 | ADVISE:ORACLE_DIAG | 0.078 | high_continuation_risk, oracle_actionable |
| a3ca7276563443fbb1a8291835219e58#t4 | 0.3636 | False | 0.366 | ADVISE:ORACLE_DIAG | 0.070 | high_continuation_risk, oracle_actionable |
| a3ca7276563443fbb1a8291835219e58#t6 | 0.5455 | False | 0.243 | REPLAN:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| a3ca7276563443fbb1a8291835219e58#t8 | 0.7273 | False | 0.077 | FINAL_AUDIT:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| a3ca7276563443fbb1a8291835219e58#t9 | 0.8182 | False | 0.083 | GENERIC:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| aaaa9f1cee444acd8e47832f4d416a8d#t10 | 0.8333 | False | 0.401 | ADVISE:ORACLE_DIAG | 0.252 | high_continuation_risk, oracle_actionable |
| aaaa9f1cee444acd8e47832f4d416a8d#t3 | 0.25 | False | 0.510 | ADVISE:ORACLE_DIAG | 0.150 | high_continuation_risk, oracle_actionable, harm_sensitive |
| aaaa9f1cee444acd8e47832f4d416a8d#t4 | 0.3333 | False | 0.560 | ADVISE:ORACLE_DIAG | 0.300 | high_continuation_risk, oracle_actionable |
| aaaa9f1cee444acd8e47832f4d416a8d#t7 | 0.5833 | False | 0.480 | ADVISE:ORACLE_DIAG | 0.209 | high_continuation_risk, oracle_actionable |
| aaaa9f1cee444acd8e47832f4d416a8d#t9 | 0.75 | False | 0.569 | ADVISE:ORACLE_DIAG | 0.423 | high_continuation_risk, oracle_actionable, harm_sensitive |
| b10b56df486a498eb3171c1405259e14#t11 | 0.55 | False | 0.307 | ADVISE:ORACLE_DIAG | 0.042 | high_continuation_risk, oracle_actionable |
| b10b56df486a498eb3171c1405259e14#t15 | 0.75 | False | 0.440 | ADVISE:ORACLE_DIAG | 0.411 | high_continuation_risk, oracle_actionable |
| b10b56df486a498eb3171c1405259e14#t18 | 0.9 | False | 0.458 | ADVISE:ORACLE_DIAG | 0.408 | high_continuation_risk, oracle_actionable |
| b10b56df486a498eb3171c1405259e14#t4 | 0.2 | False | 0.539 | ADVISE:ORACLE_DIAG | 0.178 | high_continuation_risk, oracle_actionable |
| b10b56df486a498eb3171c1405259e14#t8 | 0.4 | False | 0.567 | ADVISE:ORACLE_DIAG | 0.467 | high_continuation_risk, oracle_actionable |
| bd2441fbf2ee474199b6514dbfc404e5#t10 | 0.7692 | False | 0.097 | ADVISE:ORACLE_DIAG | 0.028 | high_continuation_risk, oracle_actionable, harm_sensitive |
| bd2441fbf2ee474199b6514dbfc404e5#t11 | 0.8462 | False | 0.252 | REPLAN:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable |
| bd2441fbf2ee474199b6514dbfc404e5#t3 | 0.2308 | False | 0.139 | VERIFY:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| bd2441fbf2ee474199b6514dbfc404e5#t5 | 0.3846 | False | 0.248 | FINAL_AUDIT:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| bd2441fbf2ee474199b6514dbfc404e5#t7 | 0.5385 | False | 0.241 | REPLAN:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable |
| be16413f06314d30b6879fd1e5bcbbf2#t11 | 0.7333 | False | 0.460 | ADVISE:ORACLE_DIAG | 0.422 | high_continuation_risk, oracle_actionable, harm_sensitive |
| be16413f06314d30b6879fd1e5bcbbf2#t13 | 0.8667 | False | 0.435 | ADVISE:ORACLE_DIAG | 0.373 | high_continuation_risk, oracle_actionable, harm_sensitive |
| be16413f06314d30b6879fd1e5bcbbf2#t3 | 0.2 | False | 0.465 | ADVISE:ORACLE_DIAG | 0.211 | high_continuation_risk, oracle_actionable |
| be16413f06314d30b6879fd1e5bcbbf2#t6 | 0.4 | False | 0.446 | ADVISE:ORACLE_DIAG | 0.328 | high_continuation_risk, oracle_actionable |
| be16413f06314d30b6879fd1e5bcbbf2#t8 | 0.5333 | False | 0.448 | ADVISE:ORACLE_DIAG | 0.267 | high_continuation_risk, oracle_actionable, harm_sensitive |
| c093bbeb623940fb9ea571c09c07a0f6#t3 | 0.3 | False | 0.520 | ADVISE:ORACLE_DIAG | 0.347 | high_continuation_risk, oracle_actionable, harm_sensitive |
| c093bbeb623940fb9ea571c09c07a0f6#t4 | 0.4 | False | 0.733 | ADVISE:ORACLE_DIAG | 0.431 | high_continuation_risk, oracle_actionable |
| c093bbeb623940fb9ea571c09c07a0f6#t5 | 0.5 | False | 0.573 | ADVISE:ORACLE_DIAG | 0.467 | high_continuation_risk, oracle_actionable |
| c093bbeb623940fb9ea571c09c07a0f6#t7 | 0.7 | False | 0.530 | ADVISE:ORACLE_DIAG | 0.397 | high_continuation_risk, oracle_actionable |
| c093bbeb623940fb9ea571c09c07a0f6#t8 | 0.8 | False | 0.506 | ADVISE:ORACLE_DIAG | 0.300 | high_continuation_risk, oracle_actionable |
| c58f1cf0114e4cadb96130a6ce7cd86f#t10 | 0.5556 | False | 0.613 | ADVISE:ORACLE_DIAG | 0.331 | high_continuation_risk, oracle_actionable |
| c58f1cf0114e4cadb96130a6ce7cd86f#t14 | 0.7778 | False | 0.451 | ADVISE:ORACLE_DIAG | 0.247 | high_continuation_risk, oracle_actionable |
| c58f1cf0114e4cadb96130a6ce7cd86f#t16 | 0.8889 | False | 0.466 | ADVISE:ORACLE_DIAG | 0.153 | high_continuation_risk, oracle_actionable, harm_sensitive |
| c58f1cf0114e4cadb96130a6ce7cd86f#t3 | 0.1667 | False | 0.550 | ADVISE:ORACLE_DIAG | 0.010 | high_continuation_risk, oracle_actionable |
| c58f1cf0114e4cadb96130a6ce7cd86f#t7 | 0.3889 | False | 0.548 | FINAL_AUDIT:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable |
| d6c804f1bf9c4a04a6a527d20d5bd46a#t10 | 0.7143 | False | 0.546 | ADVISE:ORACLE_DIAG | 0.453 | high_continuation_risk, oracle_actionable |
| d6c804f1bf9c4a04a6a527d20d5bd46a#t12 | 0.8571 | False | 0.488 | ADVISE:ORACLE_DIAG | 0.404 | high_continuation_risk, oracle_actionable, harm_sensitive |
| d6c804f1bf9c4a04a6a527d20d5bd46a#t3 | 0.2143 | False | 0.384 | ADVISE:ORACLE_DIAG | 0.300 | high_continuation_risk, oracle_actionable, harm_sensitive |
| d6c804f1bf9c4a04a6a527d20d5bd46a#t5 | 0.3571 | False | 0.440 | ADVISE:ORACLE_DIAG | 0.247 | high_continuation_risk, oracle_actionable, harm_sensitive |
| d6c804f1bf9c4a04a6a527d20d5bd46a#t8 | 0.5714 | False | 0.468 | ADVISE:ORACLE_DIAG | 0.381 | high_continuation_risk, oracle_actionable, harm_sensitive |
| e8c6d52a59884a3db4ec3f8b1c888fa7#t10 | 0.7143 | False | 0.060 | GENERIC:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| e8c6d52a59884a3db4ec3f8b1c888fa7#t12 | 0.8571 | False | -0.016 | ADVISE:TYPE_TARGET | 0.000 | high_continuation_risk, irrecoverable_under_scope, harm_sensitive |
| e8c6d52a59884a3db4ec3f8b1c888fa7#t3 | 0.2143 | False | 0.034 | GENERIC:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| e8c6d52a59884a3db4ec3f8b1c888fa7#t5 | 0.3571 | False | 0.024 | FINAL_AUDIT:TYPE_TARGET | 0.000 | high_continuation_risk, oracle_actionable, harm_sensitive |
| e8c6d52a59884a3db4ec3f8b1c888fa7#t8 | 0.5714 | False | 0.040 | ADVISE:ORACLE_DIAG | 0.002 | high_continuation_risk, oracle_actionable, harm_sensitive |
| e8da9b8f67a1499188521045f47b3150#t3 | 0.375 | False | 0.402 | ADVISE:ORACLE_DIAG | 0.030 | high_continuation_risk, oracle_actionable, harm_sensitive |
| e8da9b8f67a1499188521045f47b3150#t4 | 0.5 | False | 0.580 | ADVISE:ORACLE_DIAG | 0.132 | high_continuation_risk, oracle_actionable |
| e8da9b8f67a1499188521045f47b3150#t6 | 0.75 | False | 0.377 | ADVISE:ORACLE_DIAG | 0.308 | high_continuation_risk, oracle_actionable, harm_sensitive |

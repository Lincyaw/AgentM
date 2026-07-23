# Label validity (the scoring instrument, not the agent)

This angle does not examine the agent. It examines whether the recorded
evaluation result is a valid measurement — the gate every label must pass
before any other dimension's finding may cite it.

Work through the evaluation evidence as an auditor of the instrument: does
the recorded reward actually follow from the verifier and judge outputs? The
failure shapes are an empty or missing reward where a default silently became
the score; a judge that crashed, emitted malformed output, or never ran; a
verifier environment that failed before testing anything; and a judge whose
stated reasoning misreads the patch it was scoring. Distinguish instrument
failure (the score is noise) from instrument disagreement (the score is a
judgment call); only the former invalidates the label.

Also state the direction of the residual truth: when the instrument broke,
what does the surviving evidence say the outcome actually was?

Online signature guidance: none in the usual sense — this gate protects the
offline mining set. Its online counterpart is attribution hygiene: a live
monitor telling the agent about environment breakage so the agent does not
internalize a harness fault as its own error.

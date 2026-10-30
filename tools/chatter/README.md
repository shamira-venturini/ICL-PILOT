# CHAT validator

configs/validation/chatter.json pins the native Chatter archive/binary checksums and required validation options. The local executable is ignored by Git.

chat.chatter_runtime installs or selects the verified binary. chat.validation runs fresh read-only checks and verifies complete coverage and configuration/source hashes. The annotation wrapper independently enforces input and final-output validation.

Do not suppress diagnostics or run automatic fix/normalize commands. Chatter establishes structural validity; separate checks establish word/header/tier preservation. Neither certifies linguistic annotation accuracy.

Use standard structural and dependent-tier alignment validation. The optional
`--strict-linkers` mode is not enabled for real or synthetic CHAT: cross-utterance
quotation/completion marker relationships are outside the study outcomes. Keep
original markers and utterance boundaries. This choice does not disable parsing,
alignment, complete fresh coverage or source-preservation checks.

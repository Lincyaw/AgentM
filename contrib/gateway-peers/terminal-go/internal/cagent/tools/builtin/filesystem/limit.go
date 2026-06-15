package filesystem

const maxFiles = 100

// maxSearchFileSize bounds how large a single file may be before
// search_files_content skips it. Files above this size are almost always
// binaries, logs or data dumps that blow up memory when read whole
// (os.ReadFile + string(...) + strings.Split triples the footprint).
// Skipping them keeps a recursive search over a large tree bounded.
const maxSearchFileSize = 10 << 20 // 10 MiB

// maxBinarySniffBytes is how much of a file header search_files_content
// reads to decide whether the file is binary before loading the rest.
const maxBinarySniffBytes = 512

// maxSearchOutputBytes caps the total size of the joined match output.
// Without it a broad search (e.g. across the home directory) produces a
// multi-megabyte result that is then copied into the in-memory message
// list, written to the session store, and re-serialised into every
// subsequent model request for the rest of the session.
const maxSearchOutputBytes = 1 << 20 // 1 MiB

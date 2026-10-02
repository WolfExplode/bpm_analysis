# Recording identity is a fingerprint of the audio samples

A [[CONTEXT#Recording|Recording]] is identified by a hash of its decoded audio samples — not its filename, not a hash of the file's bytes, and not an ID written into the file. [[CONTEXT#Analysis|Analyses]] are stored under that fingerprint and every [[CONTEXT#Annotation|Annotation]] records the fingerprint of the recording it was made on; the Annotation's sidecar filename is only a convenience.

Filenames are useless as identity here because BPM tags are written into them and change whenever the algorithm (or its parameters) changes. Writing an ID into the file's metadata was rejected: it mutates source recordings — including the irreplaceable ground-truth ones — and is silently lost if a tool strips metadata chunks. Hashing the samples rather than the file bytes means renames and metadata edits never break the link; only re-encoding, resampling or trimming does, and that genuinely makes a different recording.

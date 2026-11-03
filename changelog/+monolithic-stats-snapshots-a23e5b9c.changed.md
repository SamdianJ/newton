Store internal monolithic residual and material reaction diagnostics as independent,
read-only NumPy snapshots instead of eager Python lists; use the existing recording
serializer for list-based JSON export. Retain full failure vectors in material and
cloth runner JSON diagnostics.

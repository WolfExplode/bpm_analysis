# A native Python workspace replaces the Plotly HTML reports

Results are viewed and annotated in a native desktop workspace (Python, Qt + PyQtGraph) that opens saved [[CONTEXT#Analysis|Analyses]], instead of self-contained Plotly HTML files. The HTML reports, their JavaScript annotation editor and the tkinter GUI are removed, not maintained alongside.

A static HTML file must embed every sample as JSON and render it all in the browser; hour-long recordings carry millions of points per trace, so it was either slow or had to throw data away up front. A live process can decimate to the visible window on every zoom, which is the only approach that scales. Python was chosen over a faster UI stack (Rust, or TypeScript/WebGL) because the tool is mainly used to debug the algorithm: in one language, a new debug trace is one line in the engine and appears in the viewer automatically (the Analysis carries self-describing named traces), and there is no second codebase or serialization layer to keep in step. The engine runs in a fresh worker process per analysis, so code edits are picked up on every re-run (Ctrl+R) without restarting the app; breakpoint debugging goes through the CLI. Rendering is not the bottleneck — PyQtGraph draws via Qt/OpenGL and Python only touches decimated, on-screen data.

## Consequences

- No more shareable single-file reports; exports (PNG, CSV) come from the workspace when needed.
- If the viewer ever needs to be a product for other people, the Analysis file is the boundary to rebuild it against in another stack; the engine is unaffected.

This “two-file” approach (one C++ program, one Python script) often avoids the headaches of Windows .pyd building, library paths, and linking issues—at the cost of having an extra process. For many use cases (especially if you’re not calling the C++ code thousands of times in a tight loop), it’s perfectly acceptable and much simpler to maintain.




Performance:
There’s an overhead of spawning a new process each time you call the C++ logic. For large or real-time tasks, calling an in-process C++ extension (via pybind11) might be faster.
Data Exchange:
You must pass arguments via command line (or perhaps read them from a file/JSON), and parse results from stdout or a file. This is less direct than calling a Python function in-process.
## Unit Tests for DISTIL
The contents of this folder include the unit testing framework for DISTIL. It uses Python's unittest module. Here, we ensure that various aspects of DISTIL function correctly using synthetic data.

To run a test case, make sure to be in the base folder of DISTIL; otherwise, there will likely be import errors depending on the manner of installation.

To run all unit tests:

```
python -m unittest discover test
```

To run a specific unit test:

```
python -m unittest test/rest_of_path_to_file
```
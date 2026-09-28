"""
run_tests.py - Test runner script to execute all unit and integration tests.
Can be executed with standard Python: python run_tests.py
"""

import os
import sys
import unittest

def main():
    print("=" * 60)
    print(" Running Podcast Clip Finder Test Suite ")
    print("=" * 60)

    # Discover and run tests in the 'tests' directory
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 60)

    if result.wasSuccessful():
        print("ALL TESTS PASSED SUCCESSFULLY! [OK]")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED. [FAIL]")
        sys.exit(1)

if __name__ == "__main__":
    main()

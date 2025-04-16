import unittest
from utils import get_normalized_path_from_artifact_uri


class GeneralTestCase(unittest.TestCase):

    def test_pickle_file_operations(self):
        import pickle
        import os

        tests_path = "file:///" + os.path.normpath(os.path.join(os.getcwd(), "tests")).replace(
            "\\", "/")
        print(tests_path)

        test_data = {"key": "value"}
        test_file = os.path.join(get_normalized_path_from_artifact_uri(tests_path), "dummy_test_file.pkl")

        # Check the file does not exist
        self.assertFalse(os.path.exists(test_file), "Test file should not exist initially.")


        # Save dummy pickle file
        with open(test_file, "wb") as f:
            pickle.dump(test_data, f)

        # Check the file exists
        self.assertTrue(os.path.exists(test_file), "Test file should exist after being created.")

        # Load and verify contents
        with open(test_file, "rb") as f:
            loaded_data = pickle.load(f)
        self.assertEqual(loaded_data, test_data)

        # Remove the file
        os.remove(test_file)

        # Confirm file no longer exists
        self.assertFalse(os.path.exists(test_file), "Test file should no longer exist after deletion.")



if __name__ == '__main__':
    unittest.main()

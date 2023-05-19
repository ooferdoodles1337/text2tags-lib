import os
import sys
import unittest
from unittest.mock import patch


# Add the parent directory of test_functions.py to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the functions module from the parent directory
from text2tags.functions import *

# class TestDownloadModel(unittest.TestCase):
#     def test_download_model(self):
#         # Define a temporary save path for testing
#         save_path = "test_model.bin"

#         # Download the model
#         download_model(save_path=save_path)

#         # Assert that the model file exists
#         self.assertTrue(os.path.exists(save_path))

#         # Clean up: remove the downloaded model file
#         os.remove(save_path)


class TestTaggerLlama(unittest.TestCase):
    def setUp(self):
        self.llama = TaggerLlama(
            model_path=os.path.join('models','ggml-model-q4_0.bin'), tag_list=["tag1", "tag2"])

    def test_preprocess_tag(self):
        preprocessed_tag = self.llama.preprocess_tag("(tag) Example")
        self.assertEqual(preprocessed_tag, "(tag)")

    def test_find_closest_tag(self):
        closest_tag = self.llama.find_closest_tag(
            "examples", threshold=2, tag_list=["ex", "example", "test"])
        self.assertEqual(closest_tag, "example")

    def test_correct_tags(self):
        tags = ["tag1", "(Tag) Example", "unknown"]
        corrected_tags = self.llama.correct_tags(
            tags, tag_list=["tag1", "(Tag) Example", "test"])
        self.assertEqual(corrected_tags, ["tag1", "(Tag) Example"])

    @patch.object(TaggerLlama, "create_completion")
    def test_predict_tags(self, mock_create_completion):
        mock_create_completion.return_value = {
            "choices": [
                {"text": "### Caption: Test Caption\n### Tags: tag1, tag2, tag3"}
            ]
        }

        predicted_tags = self.llama.predict_tags(prompt="Test Caption")
        self.assertEqual(predicted_tags, ["tag1", "tag2"])

    def test_load_tags(self):
        # Create a temporary tags.txt file for testing
        with open("lookups/temptags.txt", "w") as f:
            f.write("tag1\ntag2\ntag3\n")

        llama = TaggerLlama(model_path="model_path")
        self.assertEqual(llama.tag_list, ["tag1", "tag2", "tag3"])

        # Remove the temporary tags.txt file
        os.remove("lookups/temptags.txt")


if __name__ == "__main__":
    unittest.main()

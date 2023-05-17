import os
import sys
import unittest

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


class TestLoadTags(unittest.TestCase):
    def test_load_tags(self):
        # Assuming 'tags.txt' file exists in the 'lookups' directory
        expected_tags = ["1girl", "solo", "long_hair", "breasts", "blush"]

        actual_tags = load_tags()

        self.assertEqual(actual_tags[:5], expected_tags)


class TestCorrectTags(unittest.TestCase):
    def test_correct_tags(self):
        tags = [
            "1boy",
            "adjusting_hair",
            "black_bow",
            "black_vest",
            "bow",
            "bowtie",
            "brown_eyes",
            "clothed_male_nude_female",
            "flower",
            "garma_zabi",
            "gundam",
            "hair_flower",
            "male_focus",
            "pants",
            "purple_eyes",
            "short_hair",
            "smile",
            "solo",
            "suit",
            "vest",
            "young",
            "zaku_(gundam)",
        ]
        corrected_tags = [
            "1boy",
            "adjusting_hair",
            "black_bow",
            "black_vest",
            "bow",
            "bowtie",
            "brown_eyes",
            "clothed_male_nude_female",
            "flower",
            "gundam",
            "hair_flower",
            "male_focus",
            "pants",
            "purple_eyes",
            "short_hair",
            "smile",
            "solo",
            "suit",
            "vest",
        ]
        self.assertEqual(correct_tags(tags, tag_list=load_tags()), corrected_tags)


class TestTaggerLlama(unittest.TestCase):
    def setUp(self):
        # Initialize the TaggerLlama instance with necessary parameters
        self.tagger = TaggerLlama(
            model_path="models\ggml-model-q4_0.bin", tag_list=load_tags()
        )

    def test_predict_tags(self):
        # Define a test prompt
        prompt = "Minato aqua, a hololive vtuber with pink and blue streaked hair in a maid outfit"

        # Call the predict_tags method
        result = self.tagger.predict_tags(prompt)

        # Assert the result meets the expected conditions
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(tag, str) for tag in result))
        self.assertGreater(len(result), 0)
        # Add more assertions as needed


if __name__ == "__main__":
    unittest.main()

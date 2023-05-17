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


if __name__ == "__main__":
    unittest.main()

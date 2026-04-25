import unittest
import os
import tensorflow as tf
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data_processing import create_dataset

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'data/processed/test_region'
        os.makedirs(self.test_dir, exist_ok=True)
        self.img_path = os.path.join(self.test_dir, 'dummy.jpeg')
        
        dummy_image = tf.zeros([2, 2, 1], dtype=tf.uint8)
        encoded_image = tf.io.encode_jpeg(dummy_image)
        tf.io.write_file(self.img_path, encoded_image)

    def test_create_dataset(self):
        batch_size = 1
        ds = create_dataset(self.test_dir, batch_size=batch_size, img_size=(64, 64))
        
        for x, y in ds.take(1):
            self.assertEqual(x.shape, (batch_size, 64, 64, 1))
            self.assertEqual(y.shape, (batch_size, 64, 64, 1))
            self.assertEqual(x.dtype, tf.float32)
            self.assertTrue(tf.reduce_min(x) >= 0.0)
            self.assertTrue(tf.reduce_max(x) <= 1.0)

    def tearDown(self):
        if os.path.exists(self.img_path):
            os.remove(self.img_path)
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)

if __name__ == '__main__':
    unittest.main()

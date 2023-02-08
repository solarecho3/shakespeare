from dataclasses import dataclass
import modules.model as model
import unittest

class test_Data(model.Data):
    def test_init():
        assert test_Data.data, False

if __name__ == '__main__':
    unittest.main() 
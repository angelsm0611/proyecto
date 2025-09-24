import unittest
import os
from traductor import predict_  # Ajusta al nombre de tu función

class TestPredictSimplification(unittest.TestCase):
    def setUp(self):
        self.checkpoint_dir = "./checkpoints/EleutherAI_gpt-neo-125M"  # Ajusta al mejor modelo
        self.test_text = """Muscle cramps are a common problem characterized by a sudden, painful, involuntary contraction of muscle. These true cramps, which originate from peripheral nerves, may be distinguished from other muscle pain or spasm. Medical history, physical examination, and a limited laboratory screen help to determine the various causes of muscle cramps. Despite the "benign" nature of cramps, many patients find the symptom very uncomfortable. Treatment options are guided both by experience and by a limited number of therapeutic trials. Quinine sulfate is an effective medication, but the side-effect profile is worrisome, and other membrane-stabilizing drugs are probably just as effective. Patients will benefit from further studies to better define the pathophysiology of muscle cramps and to find more effective medications with fewer side-effects."""


    def test_predict_simplification(self):
        if not os.path.exists(self.checkpoint_dir):
            self.skipTest(f"Checkpoint {self.checkpoint_dir} not found. Run notebook to train.")
        
        try:
            result = predict_(self.test_text, self.checkpoint_dir)
            print(f"Generated: {result[:100]}...")  # Debug print
            self.assertIsInstance(result, str, "Result should be a string")
            self.assertTrue(len(result.strip()) > 10, "Result should not be too short")
            self.assertNotEqual(result, self.test_text, "Result should differ from input")
        except Exception as e:
            self.fail(f"Prediction failed: {e}")

if __name__ == "__main__":
    unittest.main()
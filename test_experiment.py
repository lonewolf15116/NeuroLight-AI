import unittest
import numpy as np
from experiment import Circuit, Predictor, dataset, control


class ExperimentTests(unittest.TestCase):
    def test_reproducibility_and_light_response(self):
        a, b, dark = Circuit(4), Circuit(4), Circuit(4)
        ra = [a.step(1) for _ in range(30)]
        rb = [b.step(1) for _ in range(30)]
        rd = [dark.step(0) for _ in range(30)]
        np.testing.assert_equal(ra, rb)
        self.assertGreater(np.mean(ra[-10:]), np.mean(rd[-10:])+10)

    def test_learning_and_controller_bounds(self):
        x, y = dataset(range(3), 80)
        model = Predictor(); before = np.mean((model.predict(x)-y)**2)
        model.fit(x, y, 100)
        self.assertLess(np.mean((model.predict(x)-y)**2), before/4)
        _, rates, lights, _ = control(model, 99, .5, 'learned')
        self.assertTrue(np.isfinite(rates).all())
        self.assertTrue(((lights >= 0) & (lights <= 1)).all())


if __name__ == '__main__':
    unittest.main()

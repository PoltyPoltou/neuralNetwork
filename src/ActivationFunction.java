import org.jblas.DoubleMatrix;

public class ActivationFunction {

	double apply(double value) {
		return 1 / (1 + (Math.exp(-value)));
	}

	DoubleMatrix apply(DoubleMatrix m) {
		DoubleMatrix matrix = DoubleMatrix.zeros(m.rows, m.columns);
		for (int i = 0; i < m.length; ++i) {
			matrix.put(i, apply(m.get(i)));
		}
		return matrix;
	}

	double diff(double value) {
		double x = apply(value);
		return x * (1 - x);
	}

	DoubleMatrix diff(double[] t) {
		return diff(new DoubleMatrix(t.length, 1, t));
	}

	DoubleMatrix diff(DoubleMatrix m) {
		DoubleMatrix matrix = DoubleMatrix.zeros(m.rows, m.columns);
		for (int i = 0; i < m.length; ++i) {
			matrix.put(i, diff(m.get(i)));
		}
		return matrix;
	}
}

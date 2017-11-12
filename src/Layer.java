import org.jblas.DoubleMatrix;

public class Layer {

	private DoubleMatrix values;
	private DoubleMatrix links;
	static ActivationFunction f = new ActivationFunction();
	private double learningRate = 2;
	private int learningDecay = 5;
	private int learningTimes = 0;

	public Layer(int size, int nextLayersize) {

		this.values = DoubleMatrix.zeros(size, 2);
		this.links = DoubleMatrix.rand(nextLayersize, size);
	}

	public Layer(int size, int nextLayersize, double[] values) {
		assert (size == values.length);
		this.values = new DoubleMatrix(values);
		this.links = DoubleMatrix.rand(nextLayersize, size);
	}

	public void changeWeight(DoubleMatrix deltaweight) {
		deltaweight.assertSameSize(links);
		links.addi(deltaweight.neg().mul(1));
		if (learningDecay == learningTimes) {
			learningRate -= 0.001;
			if (learningRate < 0)
				learningRate = 1;
			learningTimes = 0;
		} else
			learningTimes++;

	}

	@Override
	public String toString() {
		String str = values.toString() + ", " + links.toString();
		return str + "\n";
	}

	public void setValuesAsInput(double[] inputs) {
		setValuesAsInput(new DoubleMatrix(inputs));
	}

	public void setValuesAsInput(DoubleMatrix inputs) {
		DoubleMatrix m = DoubleMatrix.zeros(inputs.rows, 2);
		m.addiColumnVector(inputs);
		assert (m.sameSize(values));
		values = m;
	}

	public void setValues(double[] inputs) {
		setValues(new DoubleMatrix(inputs));
	}

	public void setValues(DoubleMatrix inputs) {
		assert (inputs.rows == values.rows);
		DoubleMatrix appliedValues = f.apply(inputs);
		values.putColumn(0, inputs);
		values.putColumn(1, appliedValues);
	}

	public DoubleMatrix getAppliedValues() {
		return values.getColumn(1);
	}

	public DoubleMatrix getUnappliedValues() {
		return values.getColumn(0);
	}

	public DoubleMatrix getNextOutputValues() {
		return links.mulRowVector(this.getAppliedValues().transpose());
	}

	public DoubleMatrix getNextLayerValue() {
		return links.mmul(values.getColumn(1));
	}

	public int getSize() {
		return values.rows;
	}

	public DoubleMatrix getValue() {
		return values;
	}

	public DoubleMatrix getLinks() {
		return links;
	}

}

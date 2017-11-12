import java.time.Duration;
import java.time.Instant;

import org.jblas.DoubleMatrix;

public class Network {
	private Layer[] network;
	int inputs, outputs;
	ActivationFunction f;

	public Network(int[] neuronsPerLayer) {
		f = new ActivationFunction();
		inputs = neuronsPerLayer[0];
		outputs = neuronsPerLayer[neuronsPerLayer.length - 1];
		network = new Layer[neuronsPerLayer.length];
		for (int i = 0; i + 1 < neuronsPerLayer.length; ++i) {
			network[i] = new Layer(neuronsPerLayer[i], neuronsPerLayer[i + 1]);
		}
		network[neuronsPerLayer.length - 1] = new Layer(neuronsPerLayer[neuronsPerLayer.length - 1], 0);
	}

	@Override
	public String toString() {
		String str = "";
		for (Layer l : network) {
			str += l.toString();
		}
		return str;
	}

	public String showResult() {
		return network[network.length - 1].getAppliedValues().toString();
	}

	public void forwardPropagation(double[] inputs) {
		assert (inputs.length == this.inputs);
		network[0].setValuesAsInput(inputs);
		for (int layer = 1; layer < network.length; ++layer) {
			network[layer].setValues(network[layer - 1].getNextLayerValue());
		}
	}

	public void backwardPropagation(double[] expectations) {
		int m = network.length - 1;
		assert (expectations.length == network[m].getSize());
		DoubleMatrix[][] deltaK = new DoubleMatrix[expectations.length][network.length];
		DoubleMatrix[] diffEOverWk = new DoubleMatrix[m];
		// calcul des delta d/k
		for (int d = 0; d < expectations.length; ++d) {
			deltaK[d][m] = f.diff(network[m].getUnappliedValues()).mul(network[m].getAppliedValues().get(d) - expectations[d]);
			for (int layer = m - 1; layer >= 0; --layer) {
				deltaK[d][layer] = f.diff(network[layer].getUnappliedValues()).mul(network[layer].getLinks().transpose().mmul(deltaK[d][layer + 1]));
			}
		}
		for (int k = 0; k < m; k++) {
			diffEOverWk[k] = DoubleMatrix.zeros(network[k].getLinks().getRows(), network[k].getLinks().getColumns());
			for (int d = 0; d < expectations.length; ++d) {
				diffEOverWk[k].addi(deltaK[d][k + 1].mmul(network[k].getAppliedValues().transpose()));
			}
			network[k].changeWeight(diffEOverWk[k].mul(1 / expectations.length));
		}
	}

	public void train(int loops, double[][] inputs, double[][] outputs) {
		assert (inputs.length == outputs.length);
		Instant start = Instant.now();
		int random;
		for (int i = 0; i < loops; ++i) {
			random = (int) (Math.random() * inputs.length);
			this.forwardPropagation(inputs[random]);
			this.backwardPropagation(outputs[random]);
		}
		System.out.println(Duration.between(start, Instant.now()));
	}

	public void trainBackwardSometimes(int loops, int backwardCount, double[][] inputs, double[][] outputs) {
		assert (inputs.length == outputs.length);
		Instant start = Instant.now();
		int random, backward = 0;
		for (int i = 0; i < loops; ++i) {
			random = (int) (Math.random() * inputs.length);
			this.forwardPropagation(inputs[random]);
			if (backward == backwardCount) {
				this.backwardPropagation(outputs[random]);
				backward = 0;
			} else
				++backward;
		}
		System.out.println(Duration.between(start, Instant.now()));
	}

	// does not work
	public void trainForResults(int loops, double precison, double[][] inputs, double[][] outputs) {
		DoubleMatrix[] out = new DoubleMatrix[outputs.length];
		for (int i = 0; i < outputs.length; ++i) {
			out[i] = new DoubleMatrix(outputs[i]);
		}
		int random, i = 0;
		do {
			random = (int) (Math.random() * inputs.length);
			this.forwardPropagation(inputs[random]);
			this.backwardPropagation(outputs[random]);
		} while (!this.network[network.length - 1].getAppliedValues().compare(out[random], precison));
	}

}

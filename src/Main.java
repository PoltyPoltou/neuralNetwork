import java.io.IOException;
import java.util.Scanner;

public class Main {
	public static void main(String args[]) throws IOException {
		//format off
		int[] xor = {2 ,4,1};
		int[] begin = {1,1};
		int[] end = {0};
		double[][] xorInputs = {{1,1},{0,0},{1,0},{0,1}};
		double[][] xorOutputs = {{0},{0},{1},{1}};
		//format on
		Network net = new Network(xor);
		net.trainBackwardSometimes(20000, 3, xorInputs, xorOutputs);
		Scanner s = new Scanner(System.in);
		while (true) {
			int a = Integer.parseInt(s.next());
			int b = Integer.parseInt(s.next());
		//format off
			double[] c = {a,b};
			//format on
			net.forwardPropagation(c);
			System.out.println(net.showResult());
		}
	}
}

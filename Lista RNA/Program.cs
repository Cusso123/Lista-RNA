using System;
using System.IO;
using System.Linq;

class Perceptron
{
    static Random random = new Random();
    static double[] weights;
    static double bias;
    static double learningRate = 0.01;
    static int maxEpochs = 1000;
    static double minError = 0.01;

    static void Main(string[] args)
    {
        // Características das frutas: [peso, cor (1 = laranja, 0 = maçã)]
        double[][] inputs = new double[][]
        {
            new double[] {150, 1}, // Laranja
            new double[] {130, 0}, // Maçã
            new double[] {160, 1}, // Laranja
            new double[] {120, 0}, // Maçã
            new double[] {140, 0}, // Maçã
            new double[] {155, 1}  // Laranja
        };

        // Saídas desejadas: 1 = Laranja, 0 = Maçã
        int[] outputs = { 1, 0, 1, 0, 0, 1 };

        // Normalização dos dados
        NormalizeData(inputs);

        // Inicializa pesos e bias com valores aleatórios
        InitializeWeights(inputs[0].Length);

        // Treinamento do perceptron
        TrainPerceptron(inputs, outputs);

        // Teste com novos dados
        TestPerceptron(new double[] { 148, 1 }); // Laranja esperada
        TestPerceptron(new double[] { 135, 0 }); // Maçã esperada
        TestPerceptron(new double[] { 142, 1 }); // Laranja esperada
        TestPerceptron(new double[] { 125, 0 }); // Maçã esperada
        TestPerceptron(new double[] { 152, 1 }); // Laranja esperada

        // Salvar os pesos e bias em um arquivo
        SaveModel("perceptron_model.txt");

        // Carregar o modelo salvo
        LoadModel("perceptron_model.txt");
    }

    static void NormalizeData(double[][] inputs)
    {
        for (int i = 0; i < inputs[0].Length; i++)
        {
            double max = inputs.Max(input => input[i]);
            double min = inputs.Min(input => input[i]);
            for (int j = 0; j < inputs.Length; j++)
            {
                inputs[j][i] = (inputs[j][i] - min) / (max - min);
            }
        }
    }

    static void InitializeWeights(int inputSize)
    {
        weights = new double[inputSize];
        for (int i = 0; i < inputSize; i++)
        {
            weights[i] = random.NextDouble() * 2 - 1; // Pesos entre -1 e 1
        }
        bias = random.NextDouble() * 2 - 1; // Bias entre -1 e 1
    }

    static void TrainPerceptron(double[][] inputs, int[] outputs)
    {
        for (int epoch = 0; epoch < maxEpochs; epoch++)
        {
            double totalError = 0;

            for (int i = 0; i < inputs.Length; i++)
            {
                int predicted = Predict(inputs[i]);
                int error = outputs[i] - predicted;

                // Ajusta pesos e bias
                for (int j = 0; j < weights.Length; j++)
                {
                    weights[j] += learningRate * error * inputs[i][j];
                }
                bias += learningRate * error;

                totalError += Math.Abs(error);
            }

            // Monitoramento da função de custo
            Console.WriteLine($"Época {epoch + 1}, Erro Total: {totalError}");

            if (totalError < minError)
            {
                Console.WriteLine("Treinamento completo com sucesso!");
                break;
            }
        }
    }

    static int Predict(double[] input)
    {
        double sum = bias;
        for (int i = 0; i < input.Length; i++)
        {
            sum += weights[i] * input[i];
        }
        return sum >= 0 ? 1 : 0; // Função de ativação degrau
    }

    static void TestPerceptron(double[] input)
    {
        double[] normalizedInput = new double[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            double max = input.Max();
            double min = input.Min();
            normalizedInput[i] = (input[i] - min) / (max - min);
        }

        int result = Predict(normalizedInput);
        Console.WriteLine($"Entrada: [{string.Join(", ", input)}] => Saída prevista: {(result == 1 ? "Laranja" : "Maçã")}");
    }

    static void SaveModel(string filePath)
    {
        using (StreamWriter writer = new StreamWriter(filePath))
        {
            writer.WriteLine(string.Join(",", weights));
            writer.WriteLine(bias);
        }
        Console.WriteLine("Modelo salvo com sucesso!");
    }

    static void LoadModel(string filePath)
    {
        using (StreamReader reader = new StreamReader(filePath))
        {
            weights = reader.ReadLine().Split(',').Select(double.Parse).ToArray();
            bias = double.Parse(reader.ReadLine());
        }
        Console.WriteLine("Modelo carregado com sucesso!");
    }
}

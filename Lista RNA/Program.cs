using System;
using System.IO;
using System.Linq;

class Perceptron
{
    static Random random = new Random();
    static double[] peso;
    static double bias;
    static double taxaAprendizagem = 1;
    static int maxEpocas = 20000;
    static double minErros = 0.01;

    // Variáveis para armazenar o mínimo e o máximo de cada entrada
    static double[] minValues;
    static double[] maxValues;

    static void Main(string[] args)
    {
        //[peso, cor (1 = laranja, 0 = maçã)]
        double[][] inputs = new double[][]
        {
            new double[] {150, 1}, // Laranja
            new double[] {130, 0}, // Maçã
            new double[] {160, 1}, // Laranja
            new double[] {120, 0}, // Maçã
            new double[] {140, 0}, // Maçã
            new double[] {155, 1}  // Laranja
        };

        int[] outputs = { 1, 0, 1, 0, 0, 1 };

        // Normalização dos dados
        NormalizarDados(inputs);

        // Inicializa pesos e bias com valores aleatórios
        InicializarPeso(inputs[0].Length);

        // Treinamento do perceptron
        TreinamentoPerceptron(inputs, outputs);

        // Teste com novos dados (lembrando que usamos a mesma normalização)
        TestePerceptron(new double[] { 148, 1 }); // Laranja esperada
        TestePerceptron(new double[] { 135, 0 }); // Maçã esperada
        TestePerceptron(new double[] { 142, 1 }); // Laranja esperada
        TestePerceptron(new double[] { 125, 0 }); // Maçã esperada
        TestePerceptron(new double[] { 152, 1 }); // Laranja esperada

        // Salvar os pesos e bias em um arquivo
        SalvarModelo("perceptron_model.txt");

        // Carregar o modelo salvo
        CarregarModelo("perceptron_model.txt");
    }

    static void NormalizarDados(double[][] inputs)
    {
        int inputSize = inputs[0].Length;
        minValues = new double[inputSize];
        maxValues = new double[inputSize];

        for (int i = 0; i < inputSize; i++)
        {
            maxValues[i] = inputs.Max(input => input[i]);
            minValues[i] = inputs.Min(input => input[i]);
            for (int j = 0; j < inputs.Length; j++)
            {
                inputs[j][i] = (inputs[j][i] - minValues[i]) / (maxValues[i] - minValues[i]);
            }
        }
    }

    static void NormalizarEntrada(double[] input)
    {
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = (input[i] - minValues[i]) / (maxValues[i] - minValues[i]);
        }
    }

    static void InicializarPeso(int inputSize)
    {
        peso = new double[inputSize];
        for (int i = 0; i < inputSize; i++)
        {
            peso[i] = random.NextDouble() * 2 - 1; // Pesos entre -1 e 1
        }
        bias = random.NextDouble() * 2 - 1; // Bias entre -1 e 1
    }

    static void TreinamentoPerceptron(double[][] inputs, int[] outputs)
    {
        for (int epoch = 0; epoch < maxEpocas; epoch++)
        {
            double totalError = 0;

            for (int i = 0; i < inputs.Length; i++)
            {
                int predicted = Predict(inputs[i]);
                int error = outputs[i] - predicted;

                // Ajusta pesos e bias
                for (int j = 0; j < peso.Length; j++)
                {
                    peso[j] += taxaAprendizagem * error * inputs[i][j];
                }
                bias += taxaAprendizagem * error;

                totalError += Math.Abs(error);
            }

            Console.WriteLine($"Época {epoch + 1}, Erro Total: {totalError}");

            if (totalError < minErros)
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
            sum += peso[i] * input[i];
        }
        return sum >= 0 ? 1 : 0; // Função de ativação degrau
    }

    static void TestePerceptron(double[] input)
    {
        NormalizarEntrada(input);

        int result = Predict(input);
        Console.WriteLine($"Entrada: [{string.Join(", ", input)}] => Saída prevista: {(result == 1 ? "Laranja" : "Maçã")}");
    }

    static void SalvarModelo(string filePath)
    {
        using (StreamWriter writer = new StreamWriter(filePath))
        {
            writer.WriteLine(string.Join(",", peso));
            writer.WriteLine(bias);
        }
        Console.WriteLine("Modelo salvo com sucesso!");
    }

    static void CarregarModelo(string filePath)
    {
        using (StreamReader reader = new StreamReader(filePath))
        {
            peso = reader.ReadLine().Split(',').Select(double.Parse).ToArray();
            bias = double.Parse(reader.ReadLine());
        }
        Console.WriteLine("Modelo carregado com sucesso!");
    }
}

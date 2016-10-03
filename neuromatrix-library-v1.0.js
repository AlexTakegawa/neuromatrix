/*
Modular Library of Artificial Neural Networks
Biblioteca Modular de Redes Neurais Artificiais

The user has the possibility to use each part of a ANN, or use a preset ANN
O usuario tem a possibilidade de usar cada parte de uma RNA, ou usar um pre-definicao de uma RNA

This algorithm does not have guarantee to work, because it is an essay, made by an student, to other students learn from it
Esse algoritmo nao tem garantia de funcionar, porque e um ensaio, feito por um aluno, para outros alunos aprenderem a partir dele

Index:

0. Matrix Functions
1. Other Matrix Functions
2. Special Matrix Functions
3. Transfer and Basic Activation Functions
4. Train Methods Functions
5. Backpropagation Function
6. Hopfield Energy Core Function
7. Presets Artificial Neural Networks Functions

Indice:

0. Funcoes de Matrizes
1. Outras Funcoes de Matrizes
2. Funcoes de Matrizes Especiais
3. Funcao de Transferencia e Funcoes Basicas de Ativacao
4. Funcoes de Metodos de Treino
5. Funcao de Retropropagacao [Backpropagation]
6. Funcao do Nucleo da Funcao de Energia de Hopfield
7. Funcoes de Redes Neurais Artificiais Pre-Definidas

*/

/// 0. Matrix Functions
/// 0. Funcoes de Matrizes

/// Ordinary Matrix Addition
/// Adicao Comum de Matrizes
function matrixAddition(matrix, matrix2) {
  var newMatrix = [] ;
  for (i = 0; i < matrix.length; i++) {
	newMatrix[i] = [] ;
    for (j = 0; j < matrix2[0].length; j++) {
   	  var calculus = 0 ;
	  for (k = 0; k < matrix[0].length; k++) {
      calculus = matrix[i][j] + matrix2[i][j] ;
	  }
	  newMatrix[i][j] = calculus ;
    }
  }
  return newMatrix
}

/// Ordinary Matrix Subtraction
/// Subtracao Comum de Matrizes
function matrixSubtraction(matrix, matrix2) {
  var newMatrix = [] ;
  for (i = 0; i < matrix.length; i++) {
	newMatrix[i] = [] ;
    for (j = 0; j < matrix2[0].length; j++) {
   	  var calculus = 0 ;
	  for (k = 0; k < matrix[0].length; k++) {
        calculus =  matrix2[i][j] - matrix[i][j] ;
	  }
	  newMatrix[i][j] = calculus ;
    }
  }
  return newMatrix
}

/// Ordinary Matrix Multiplication
/// Multiplicacao Comum de Matrizes
function matrixMultiplication(matrix, matrix2) {
  var newMatrix = [] ;
  for (i = 0; i < matrix.length; i++) {
	newMatrix[i] = [] ;
    for (j = 0; j < matrix2[0].length; j++) {
   	  var calculus = 0 ;
	  for (k = 0; k < matrix[0].length; k++) {
        calculus = matrix[i][j] * matrix2[i][j] ;
	  }
	  newMatrix[i][j] = calculus ;
    }
  }
  return newMatrix
}

/// Ordinary Matrix Division
/// Divisao Comum de Matrizes
function matrixDivision(matrix, matrix2) {
  var newMatrix = [] ;
  for (i = 0; i < matrix.length; i++) {
	newMatrix[i] = [] ;
    for (j = 0; j < matrix2[0].length; j++) {
   	  var calculus = 0 ;
	  for (k = 0; k < matrix[0].length; k++) {
        calculus = matrix2[i][j] / matrix[i][j] ;
	  }
	  newMatrix[i][j] = calculus ;
    }
  }
  return newMatrix
}

/// ---------------------------------------------------------------

/// 1. Other Matrix Functions
/// 1. Outras Funcoes de Matrizes

/// Multiply Matrix by Scalar
/// Multiplicacao de uma Matriz por um Escalar
function matrixScalarAddition(matrix, scalar) {
  var newMatrix = [] ;
  for (i = 0; i < matrix.length; i++) {
	newMatrix[i] = [] ;
    var calculus = 0 ;
    for (j = 0; j < matrix[0].length; j++) {
      calculus = matrix[i][j] * scalar ;
  	  newMatrix[i][j] = calculus ;
    }
  }
  return newMatrix
}

/// Matrix Transposition
/// Transposicao de Matriz
function transposeMatrix(matrix) {
  for (i = 0; i < matrix.length; i++) {
	for (j = 0; j < i; j++) {
      var temp = matrix[i][j] ;
	  matrix[i][j] = matrix[j][i] ;
	  matrix[j][i] = temp ;
    }
  }
  return matrix
}

/// Horizontal Matrix Concatenation
/// Concatenacao de Matriz Horizontal
function concatenateMatrix(matrix, matrix2) {
  var newMatrix = matrix.concat(matrix2) ;
  return newMatrix ;
}

/// ---------------------------------------------------------------

/// 2. Special Matrix Functions
/// 2. Funcoes de Matrizes Especiais

/// Hadamard Product
/// Produto de Hadamard
/// Must be two matrix of the same size
/// Tem que ser duas matrizes do mesmo tamanho
function hadamard(matrix, matrix2) {
  var newMatrix = [] ;
  for (i = 0; i < matrix.length; i++) {
   	newMatrix[i] = [] ;
    for (j = 0; j < matrix2[0].length; j++) {
     	var calculus = 0 ;
  	  for (k = 0; k < matrix[0].length; k++) {
        calculus = matrix[i][j] * matrix2[i][j] ;
	  }
	  newMatrix[i][j] = calculus ;
	}
  }
  return newMatrix
}

/// Kronecker Product
/// Produto de Kronecker
/// Must be two matrix of the same size also
/// Tem que ser duas matrizes do mesmo tamanho tambem
function kronecker(matrix, matrix2) {
  var newMatrix = [] ;
  for (i = 0; i < matrix.length; i++) {
    newMatrix[i] = [] ;
    for (j = 0; j < matrix2[0].length; j++) {
      var calculus = 0 ;
      for (k = 0; k < matrix[0].length; k++) {
        calculus = matrix[0][i] * matrix2[j][0] ;
  	  }
  	  newMatrix[i][j] = calculus ;
  	}
  }
  return newMatrix
}

/// ---------------------------------------------------------------

/// 3. Transfer and Basic Activation Functions
/// 3. Funcao de Transferencia e Funcoes Basicas de Ativacao

/// Transfer Function
/// Funcao de Transferencia
function feedForward(inputs, weights) { /// Acho que OK!
  var sum = 0 ;
  for (i = 0; i < weights.length; i++) {
    sum += inputs[i] * weights[i] ;
  }
  return sum
}

/// Activation Functions
/// Funcoes de Ativacao

/// Identity
/// Identidade
function identity(inputFunction) {
  var output = 0 ;
  var identityFunc = inputFunction ;
  output = identityFunc ;
  return output ;
}

/// Sigmoid
/// Sigmoide
function sigmoid(inputFunction) {
  var output = 0 ;
  var sigmoidFunc = 1 - (1 / (1 + Math.pow(Math.E, inputFunction) ) ) ;
  output = sigmoidFunc
  return output ;
}

/// Hard Limiter
/// Limitador Rigido
function hardLimiter(inputFunction) {
  var hardLimiterFunc = 0;
  var output = 0 ;
  if (inputFunction >= 0) {
    hardLimiterFunc = 1 ;
  } else {
    hardLimiterFunc = 0 ;
  }
  output = hardLimiterFunc ;
  return output ;
}

/// Gaussian
/// Gaussiana
function gaussian(inputFunction, mi, sigma) {
  var output = 0 ;
  var gaussianFunc = 1 / ( sigma * Math.sqrt(2 * Math.PI) ) * Math.pow(Math.E, - Math.pow((inputFunction - mi), 2) / (2 * Math.pow(sigma, 2)) ) ;
  output = gaussianFunc ;
  return output ;
}

/// Hyperbolic Tangent
/// Tangente Hiperbolica
function hyperbolicTangent(inputFunction) {
  var hyperbolicTangentFunc = ( Math.pow( Math.E, (2 * inputFunction) ) - 1 ) / ( Math.pow( Math.E, (2 * inputFunction) ) + 1 ) ;
  output = hyperbolicTangentFunc ;
  return output ;
}

/// ---------------------------------------------------------------

/// 4. Train Methods Functions
/// 4. Funcoes de Metodos de Treino

/// Supervised
/// Supervisionado

/// Error Correction
/// Correcao de Erros
function errorCorrection(feedForwardFunction, inputs,  weights, desiredOutput, learningRate) {
  var guess = feedForwardFunction ;
  var error = desiredOutput - guess ;
  var errorCorrectionLearning = 0 ;
  for (i = 0; i < weights.length; i++) {
    errorCorrectionLearning += learningRate * error * inputs ;
  }
  return errorCorrectionLearning
}

/// Unsupervised
/// Não Supervisionado

/// Hebbian
/// Hebbiana
function hebbian(inputFunction, learningRate) {
  var hebbianLearning = [] ;
  for (i = 0; i <= inputFunction.length - 1; i++) {
    hebbianLearning[i] = [] ;
    for (j = 0; j <= inputFunction[0].length - 1; j++) {
      hebbianLearning[i][j] = learningRate * inputFunction[i][j] ;
    }
  }
  return hebbianLearning
}

/// Euclidian Distance Function [For The Kohonen's Self-Organizing Maps]
/// Funcao de Distancia Euclidiana [Para Os Mapas Auto-Organizaveis de Kohonen]
function euclidianDistance(x, y) {
  var output = 0;
  for (i = 0; i <= x.length - 1; i++) {
    for (j = 0; j <= y.length - 1; j++) {
      output = Math.sqrt(Math.pow((y[j] - x[i]), 2)) ;
    }
  }
  return output
}

/// Competitive
/// Competitivo
function competitive(inputFunction, weights, learningRate) {
  var competitiveLearning = 0 ;
  for (i = 0; i <= inputFunction.length - 1; i++) {
    competitiveLearning[i] = [] ;
    for (j = 0; j <= inputFunction[0].length - 1; j++) {
      var bestWeight = euclidianDistance([inputFunction[i][j]], [weights[i][j]]) ;
      weights[i][j] += learningRate * inputFunction[i][j] ;
      competitiveLearning = weights[i][j] / bestWeight ;
    }
  }
  return competitiveLearning
}

/// Kohonen's Self-Organizing Maps
/// Mapas Auto-Organizaveis de Kohonen
function kohonen(inputFunction, weights, learningRate, neighborhood) {
  var bmu = 0 ;
  var kohonenLearning = [] ;
  for (i = 0; i <= weights.length - 1; i++) {
    kohonenLearning[i] = [] ;
    for (j = 0; j <= weights[0].length - 1; j++) {
      bmu += Math.min(euclidianDistance([inputFunction[i][j]], [weights[i][j]])) ;
      kohonenLearning += bmu * learningRate * (inputFunction[i][j] - weights[i][j]) ;
    }
  }
  return kohonenLearning
}

/// ---------------------------------------------------------------

/// 5. Backpropagation Function
/// 5. Funcao de Retropropagacao [Backpropagation]

/// Backpropagation
/// Retropropagacao
function backpropagation(feedForwardFunction, weights, learningRate) {
  var correctValue = 0 ;
  var oj = 0 ;
  var phi = 0 ;
  var outputDelta = 0 ;
  var deltaWeight = 0 ;
  var errors = 0 ;
  for (i = 0; i < weights.length - 1; i++){
    for (j = 0; j < weights[0].length - 1; j++){
      var prediction = feedForwardFunction ;
      oj = sigmoid(feedForwardFunction) ;
      outputDelta = oj * (1 - oj) ;
      deltaWeight = - learningRate * outputDelta * feedForwardFunction ;
      weights[i][j] += weights[i][j] * deltaWeight ;

      if (weights[i][j] == prediction) {
        errors = - learningRate * feedForwardFunction * (oj - correctValue) * oj * (1 - oj) ;
      } else {
        errors = - learningRate * feedForwardFunction * (outputDelta - weights[i][j]) * oj * (1 - oj) ;
      }

    }
    return errors
  }
}



/// ---------------------------------------------------------------

/// 6. Hopfield Energy Core Function
/// 6. Funcao do Nucleo da Funcao de Energia de Hopfield

/// Hopfield Artificial Neural Network Core
/// Nucleo da Rede Neural Artificial de Hopfield
function hopfieldCore(inputs, weights, nodes, learningRate, bias) {
  for (i = 1; i <= nodes; i++) {
	var hopfieldCoreFunc = 0 ;
    hopfieldCoreFunc = weights[i] * sigmoid(learningRate - bias) + inputs[i] ;
  }
  return hopfieldCoreFunc
}

/// ---------------------------------------------------------------

/// 7. Presets Artificial Neural Networks Functions
/// 7. Funcoes de Redes Neurais Artificiais Pre-Definidas

/// Single Layer Perceptron
/// Perceptron de Unica Camada
function perceptron(inputs, desiredOutput, weights, learningRate) {
  var inputFunction = feedForward(inputs, weights) ;
  var binaryOutput = hardLimiter(inputFunction) ;
  return binaryOutput
}

/// Standard Hopfield Artificial Neural Network
/// Rede Neural Artificial de Hopfield Padrao
function hopfield(inputs, weights, nodes, learningRate, bias) {
  var networkCore = hopfieldCore(inputs, weights, nodes, learningRate, bias) ;
  var binaryOutput = hardLimiter(networkCore) ;
  return binaryOutput
}

/// Continuous Time Hopfield Artificial Neural Network
/// Rede Neural Artificial Hopfield de Tempo Continuo
function continuousTimeHopfield(inputs, weights, nodes, learningRate, bias) {
  var cth = -1/2 + hopfieldCore(inputs, weights, nodes, learningRate, bias) ;
  var binaryOutput = hardLimiter(cth) ;
  return binaryOutput 
}

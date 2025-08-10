const { vocab, data, embedding, wk, wq, wv, w1, w2, b1, b2, W, B } = require('./data')

function positionalEncoding(numPositions, dModel) {
    return Array.from({ length: numPositions }, (_, pos) => {
        const PE = [];
        for (let i = 0; i < dModel; i++) {
            const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / dModel);
            PE.push(i % 2 === 0 ? Math.sin(angle) : Math.cos(angle));
        }
        return PE;
    });
}

pe = positionalEncoding(3, 4);

const final = data.map((row, i) => row.map((cols, j) => cols + pe[i][j]))

function matMul(a, b) {
    const row = [];
    for (let i = 0; i < a.length; i++) {
        row[i] = [];
        for (let j = 0; j < b[0].length; j++) {
            let sum = 0;
            for (let k = 0; k < a[0].length; k++) {
                sum += a[i][k] * b[k][j];
            }
            row[i][j] = sum;
        }
    }
    return row;
}

const Q = matMul(final, wq);
const K = matMul(final, wk);

const K_T = K[0].map((_, i) => K.map(row => row[i]));
const V = matMul(final, wv);

// console.log(Q)
// console.log("--------")
// console.log(K)
// console.log('---------')
// console.log(K_T)

const fff = matMul(Q, K_T)

// console.log(fff)

function mask() {
    return Array.from({ length: 3 }, (_, i) => {
        const ff = [];
        for (let j = 0; j < 3; j++) {
            ff.push(i < j ? -Infinity : 0)
        }
        return ff;
    })
}

const maskdata = mask();

const dd = fff.map((row, i) => row.map((cols, j) => cols + maskdata[i][j]));

// console.log(dd)

const ggg = dd.map((row, i) => row.map((cols, j) => cols / Math.sqrt(4)))

function softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b);
    return exps.map(e => e / sum);
}

const jgsfg = ggg.map(softmax)

const fin = matMul(jgsfg, V)

// console.log(fin)

// console.log(fin)

function layerNorm(vec, epsilon = 1e-6) {
  const mean = vec.reduce((sum, val) => sum + val, 0) / vec.length;
  const variance = vec.reduce((sum, val) => sum + (val - mean) ** 2, 0) / vec.length;
  const norm = vec.map(val => (val - mean) / Math.sqrt(variance + epsilon));
  return norm;
}

function addAndLayerNorm(x, sublayerOutput) {
  const added = x.map((row, i) => row.map((val, j) => val + sublayerOutput[i][j]));
  return added.map(data => layerNorm(data));
}

const hf = addAndLayerNorm(final, fin)


// console.log(hf)

function feedForward(matrix) {

    const out1 = addB1(matMul(matrix, w1), b1).map((row) => row.map(relu))
    return addB1(matMul(out1, w2), b2);
}

function addB1(matrix, bias){
    return matrix.map((data) => data.map((da, i) => da + bias[i]))
}

function relu(x) {
    return Math.max(0, x);
}

const gghf = feedForward(hf);

// console.log(gghf)

const X_L = addAndLayerNorm(hf, gghf)

// console.log(finlayer)

const W_T = W[0].map((_, i) => W.map(row => row[i]));

const laye = (matMul(X_L, W_T));

const finlay = laye.map(row => {
    return row.map((data, i) => data + B[i])
})

const softmax_output = final.map(data => softmax(data))

// console.log(te)

function predictNextToken(probabilities) {
  return probabilities.map(row => row.indexOf(Math.max(...row))); // Find index of max probability in each row
}

// Predict the next tokens (greedy approach)
const nextTokens = predictNextToken(softmax_output);

console.log("Next tokens (indices):", nextTokens);
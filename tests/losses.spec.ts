import {core as mx, nn} from '..';
import {assertArrayAllTrue} from './utils';
import {assert} from 'chai';

describe('losses', () => {
  it('crossEntropy', () => {
    let logits = mx.array([[0.0, -Infinity], [-Infinity, 0.0]]);
    let indices = mx.array([0, 1], mx.int32);
    let expected = mx.array([0.0, 0.0]);
    let loss = nn.losses.crossEntropy(logits, indices);
    assertArrayAllTrue(mx.allclose(loss, expected));

    let probs = mx.array([[1.0, 0.0], [0.0, 1.0]]);
    loss = nn.losses.crossEntropy(logits, probs);
    assertArrayAllTrue(mx.isnan(loss));

    logits = mx.array([[2.0, -1.0], [-1.0, 2.0]]);
    indices = mx.array([0, 1], mx.int32);
    const weights = mx.array([1.0, 2.0]);
    expected = mx.array([0.04858735, 0.0971747]);
    loss = nn.losses.crossEntropy(logits, indices, weights);
    assertArrayAllTrue(mx.isclose(loss, expected));

    probs = mx.array([[1.0, 0.0], [0.0, 1.0]]);
    loss = nn.losses.crossEntropy(logits, probs, weights);
    assertArrayAllTrue(mx.isclose(loss, expected));

    logits = mx.array([[2.0, -1.0], [-1.0, 2.0]]);
    indices = mx.array([0, 1], mx.int32);
    expected = mx.array([0.498587, 0.498587]);
    loss = nn.losses.crossEntropy(logits, indices, undefined, undefined, 0.3);
    assertArrayAllTrue(mx.isclose(loss, expected));

    probs = mx.array([[1.0, 0.0], [0.0, 1.0]]);
    loss = nn.losses.crossEntropy(logits, probs, undefined, undefined, 0.3);
    assertArrayAllTrue(mx.isclose(loss, expected));

    logits = mx.array([[2.0, -1.0], [-1.0, 2.0]]);
    indices = mx.array([0, 1], mx.int32);
    expected = mx.array([0.49858734, 0.9971747]);
    loss = nn.losses.crossEntropy(logits, indices, weights, undefined, 0.3);
    assertArrayAllTrue(mx.isclose(loss, expected));

    probs = mx.array([[1.0, 0.0], [0.0, 1.0]]);
    loss = nn.losses.crossEntropy(logits, probs, weights, undefined, 0.3);
    assertArrayAllTrue(mx.isclose(loss, expected));
  });

  describe('binaryCrossEntropy', () => {
    it('logitsAsInputs', () => {
      const logits = mx.array([0.105361, 0.223144, 1.20397, 0.916291]);
      const targets = mx.array([0, 0, 1, 1], mx.int32);

      const lossesNone = nn.losses.binaryCrossEntropy(logits, targets, undefined, undefined, 'none');
      const expectedNone = mx.array([0.747215, 0.810930, 0.262365, 0.336472]);
      assertArrayAllTrue(mx.allclose(lossesNone, expectedNone));

      const lossesMean = nn.losses.binaryCrossEntropy(logits, targets, undefined, undefined, 'mean');
      const expectedMean = expectedNone.mean();
      assert.equal(lossesMean.item(), expectedMean.item());

      const lossesSum = nn.losses.binaryCrossEntropy(logits, targets, undefined, undefined, 'sum');
      const expectedSum = expectedNone.sum();
      assert.equal(lossesSum.item(), expectedSum.item());

      const weights = mx.array([1.0, 2.0, 1.0, 2.0]);
      const expected = mx.array([0.747215, 1.62186, 0.262365, 0.672944]);
      const loss = nn.losses.binaryCrossEntropy(logits, targets, weights, undefined, 'none');
      assertArrayAllTrue(mx.allclose(loss, expected));
    });

    it('probsAsInputs', () => {
      const probs = mx.array([0.5, 0.6, 0.7, 0.8]);
      const targets = mx.array([0, 0, 1, 1]);

      const lossesNone = nn.losses.binaryCrossEntropy(probs, targets, undefined, false, 'none');
      const expectedNone = mx.array([0.693147, 0.916291, 0.356675, 0.223144]);
      assertArrayAllTrue(mx.allclose(lossesNone, expectedNone));

      const lossesMean = nn.losses.binaryCrossEntropy(probs, targets, undefined, false, 'mean');
      const expectedMean = expectedNone.mean();
      assertArrayAllTrue(mx.allclose(lossesMean, expectedMean));

      const lossesSum = nn.losses.binaryCrossEntropy(probs, targets, undefined, false, 'sum');
      const expectedSum = expectedNone.sum();
      assertArrayAllTrue(mx.allclose(lossesSum, expectedSum));
    });

    it('tinyProbsAsInputs', () => {
      const TINY_PROB = 1e-59;
      const probs = mx.array([0, TINY_PROB, 1 - TINY_PROB, 1]);
      const targets = mx.array([0, 0, 1, 1]);

      const lossesNone = nn.losses.binaryCrossEntropy(
        probs, targets, undefined, false, 'none'
      );
      const expectedNone = mx.array([0.0, TINY_PROB, TINY_PROB, 0.0]);
      assertArrayAllTrue(mx.allclose(lossesNone, expectedNone));

      // Test with reduction 'mean'
      const lossesMean = nn.losses.binaryCrossEntropy(
        probs, targets, undefined, false, 'mean'
      );
      const expectedMean = mx.mean(expectedNone);
      assertArrayAllTrue(mx.allclose(lossesMean, expectedMean));

      // Test with reduction 'sum'
      const lossesSum = nn.losses.binaryCrossEntropy(
        probs, targets, undefined, false, 'sum'
      );
      const expectedSum = mx.sum(expectedNone);
      assertArrayAllTrue(mx.allclose(lossesSum, expectedSum));
    });
  });

  it('l1Loss', () => {
    const predictions = mx.array([0.5, 0.2, 0.9, 0.0]);
    const targets = mx.array([0.5, 0.2, 0.9, 0.0]);

    const expectedNone = mx.array([0, 0, 0, 0], mx.float32);
    const expectedSum = mx.sum(expectedNone);
    const expectedMean = mx.mean(expectedNone);

    let losses = nn.losses.l1Loss(predictions, targets, 'none');
    assert.deepEqual(losses.tolist(), expectedNone.tolist());

    losses = nn.losses.l1Loss(predictions, targets, 'sum');
    assert.deepEqual(losses.tolist(), expectedSum.tolist());

    losses = nn.losses.l1Loss(predictions, targets, 'mean');
    assert.deepEqual(losses.tolist(), expectedMean.tolist());
  });

  it('mseLoss', () => {
    const predictions = mx.array([0.5, 0.2, 0.9, 0.0]);
    const targets = mx.array([0.7, 0.1, 0.8, 0.2]);

    const expectedNone = mx.array([0.04, 0.01, 0.01, 0.04]);
    const expectedMean = mx.mean(expectedNone);
    const expectedSum = mx.sum(expectedNone);

    const lossesNone = nn.losses.mseLoss(predictions, targets, 'none');
    assertArrayAllTrue(mx.isclose(lossesNone, expectedNone));

    const lossesMean = nn.losses.mseLoss(predictions, targets, 'mean');
    assert.deepEqual(lossesMean.item(), expectedMean.item());

    const lossesSum = nn.losses.mseLoss(predictions, targets, 'sum');
    assert.deepEqual(lossesSum.item(), expectedSum.item());
  });

  it('smoothL1Loss', () => {
    const predictions = mx.array([1.5, 2.5, 0.5, 3.5]);
    const targets = mx.array([1.0, 2.0, 0.5, 2.5]);
    const beta = 1.0;

    const expectedNone = mx.array([0.125, 0.125, 0.0, 0.5]);
    const expectedSum = mx.sum(expectedNone);
    const expectedMean = mx.mean(expectedNone);

    const lossNone = nn.losses.smoothL1Loss(predictions, targets, beta, 'none');
    assertArrayAllTrue(mx.equal(lossNone, expectedNone));

    const lossSum = nn.losses.smoothL1Loss(predictions, targets, beta, 'sum');
    assert.deepEqual(lossSum.item(), expectedSum.item());

    const lossMean = nn.losses.smoothL1Loss(predictions, targets, beta, 'mean');
    assert.deepEqual(lossMean.item(), expectedMean.item());
  });

  it('nllLoss', () => {
    const logits = mx.array([[0.0, -Infinity], [-Infinity, 0.0]]);
    const targets = mx.array([0, 1], mx.int32);

    const lossesNone = nn.losses.nllLoss(logits, targets, undefined, 'none');
    const expectedNone = mx.array([0.0, 0.0]);
    assertArrayAllTrue(mx.arrayEqual(lossesNone, expectedNone));

    const lossesMean = nn.losses.nllLoss(logits, targets, undefined, 'mean');
    const expectedMean = mx.mean(expectedNone);
    assert.deepEqual(lossesMean.item(), expectedMean.item());

    const lossesSum = nn.losses.nllLoss(logits, targets, undefined, 'sum');
    const expectedSum = mx.sum(expectedNone);
    assert.deepEqual(lossesSum.item(), expectedSum.item());
  });

  it('gaussianNllLoss', () => {
    const inputs = mx.array([[0.1, 0.2], [0.3, 0.4]]);
    const targets = mx.array([[0.2, 0.1], [0.1, 0.2]]);
    const vars = mx.array([[0.1, 0.2], [0.3, 0.4]]);

    let lossesNone = nn.losses.gaussianNllLoss(inputs, targets, vars, false, undefined, 'none');
    const expectedNone = mx.array([[-1.101293, -0.779719], [-0.535320, -0.408145]]);
    assertArrayAllTrue(mx.isclose(lossesNone, expectedNone));

    const lossesMean = nn.losses.gaussianNllLoss(inputs, targets, vars, false, undefined, 'mean');
    const expectedMean = mx.mean(expectedNone);
    assertArrayAllTrue(mx.isclose(lossesMean, expectedMean));

    const lossesSum = nn.losses.gaussianNllLoss(inputs, targets, vars, false, undefined, 'sum');
    const expectedSum = mx.sum(expectedNone);
    assertArrayAllTrue(mx.isclose(lossesSum, expectedSum));

    lossesNone = nn.losses.gaussianNllLoss(inputs, targets, vars, true, undefined, 'none');
    const expectedNoneFull = mx.array([[-0.182354, 0.139220], [0.383619, 0.510793]]);
    assertArrayAllTrue(mx.isclose(lossesNone, expectedNoneFull));

    const lossesMeanFull = nn.losses.gaussianNllLoss(inputs, targets, vars, true, undefined, 'mean');
    const expectedMeanFull = mx.mean(expectedNoneFull);
    assertArrayAllTrue(mx.isclose(lossesMeanFull, expectedMeanFull));

    const lossesSumFull = nn.losses.gaussianNllLoss(inputs, targets, vars, true, undefined, 'sum');
    const expectedSumFull = mx.sum(expectedNoneFull);
    assertArrayAllTrue(mx.isclose(lossesSumFull, expectedSumFull));
  });

  it('klDivLoss', () => {
    const pLogits = mx.log(mx.array([[0.5, 0.5], [0.8, 0.2]]));
    const qLogits = mx.log(mx.array([[0.5, 0.5], [0.2, 0.8]]));

    const lossesNone = nn.losses.klDivLoss(pLogits, qLogits, undefined, 'none');
    const expectedNone = mx.array([0.0, 0.831777]);
    assertArrayAllTrue(mx.isclose(lossesNone, expectedNone));

    const lossesMean = nn.losses.klDivLoss(pLogits, qLogits, undefined, 'mean');
    const expectedMean = mx.mean(expectedNone);
    assertArrayAllTrue(mx.isclose(lossesMean, expectedMean));

    const lossesSum = nn.losses.klDivLoss(pLogits, qLogits, undefined, 'sum');
    const expectedSum = mx.sum(expectedNone);
    assertArrayAllTrue(mx.isclose(lossesSum, expectedSum));
  });

  it('tripletLoss', () => {
    const anchors = mx.array([[1, 2, 3], [1, 2, 3]]);
    const positives = mx.array([[4, 5, 6], [0, -1, 2]]);
    const negatives = mx.array([[7, 8, 9], [3, 2, 3]]);

    const tripletLoss = (reduction) => nn.losses.tripletLoss(anchors, positives, negatives, undefined, undefined, undefined, undefined, reduction);

    let lossesNone = tripletLoss('none');
    let expectedNone = mx.array([0, 2.31662]);
    assertArrayAllTrue(mx.isclose(lossesNone, expectedNone));

    let lossesMean = tripletLoss('mean');
    let expectedMean = mx.mean(expectedNone);
    assertArrayAllTrue(mx.isclose(lossesMean, expectedMean));

    let lossesSum = tripletLoss('sum');
    let expectedSum = mx.sum(expectedNone);
    assertArrayAllTrue(mx.isclose(lossesSum, expectedSum));
  });

  it('hingeLoss', () => {
    const inputs = mx.ones([2, 4]);
    const targets = mx.zeros([2, 4]);
    const loss = nn.losses.hingeLoss(inputs, targets, 'mean');
    assert.equal(loss.item(), 1.0);
  });

  it('huberLoss', () => {
    const inputs = mx.ones([2, 4]);
    const targets = mx.zeros([2, 4]);
    const loss = nn.losses.huberLoss(inputs, targets, undefined, 'mean');
    assert.equal(loss.item(), 0.5);
  });

  it('logCoshLoss', () => {
    const inputs = mx.ones([2, 4]);
    const targets = mx.zeros([2, 4]);
    const loss = nn.losses.logCoshLoss(inputs, targets, 'mean');
    assert.closeTo(loss.item() as number, 0.433781, 1e-6);
  });

  it('marginRankingLoss', () => {
    const inputs1 = mx.array([-0.573409, -0.765166, -0.0638]);
    const inputs2 = mx.array([0.75596, 0.225763, 0.256995]);
    const targets = mx.array([1, 1, -1], mx.int32);

    const losses = nn.losses.marginRankingLoss(inputs1, inputs2, targets, undefined, 'none');
    const expected = mx.array([1.329369, 0.990929, 0.0]);
    assertArrayAllTrue(mx.isclose(losses, expected));

    const lossesWithMargin = nn.losses.marginRankingLoss(inputs1, inputs2, targets, 0.5, 'none');
    const expectedWithMargin = mx.array([1.829369, 1.490929, 0.179205]);
    assertArrayAllTrue(mx.isclose(lossesWithMargin, expectedWithMargin));
  });
});

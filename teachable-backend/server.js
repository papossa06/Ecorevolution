const express = require('express');
const cors = require('cors');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 10000;

app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Cargar modelo
const MODEL_PATH = path.join(__dirname, 'model');
let model;

console.log('Cargando modelo desde:', MODEL_PATH);
tf.loadLayersModel(`file://${MODEL_PATH}/model.json`)
  .then(m => {
    model = m;
    console.log('✅ Modelo cargado');
  })
  .catch(err => {
    console.error('❌ Error al cargar modelo:', err);
  });

const upload = multer({ limits: { fileSize: 5 * 1024 * 1024 } });

app.post('/predict', upload.single('image'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No image sent' });
  if (!model) return res.status(500).json({ error: 'Model not ready' });

  try {
    const imgBuffer = await sharp(req.file.buffer)
      .resize(224, 224)
      .jpeg()
      .toBuffer();

    const decoded = tf.node.decodeImage(imgBuffer, 3);
    const tensor = tf.div(tf.expandDims(decoded), 255.0);
    const predictions = model.predict(tensor);
    const scores = await predictions.data();
    const classIndex = scores.indexOf(Math.max(...scores));

    decoded.dispose();
    tensor.dispose();
    predictions.dispose();

    res.json({ class: classIndex });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Prediction failed' });
  }
});

app.get('/', (req, res) => {
  res.json({ status: 'OK', message: 'Teachable backend ready' });
});

app.listen(PORT, () => {
  console.log(`🚀 Servidor en puerto ${PORT}`);
});
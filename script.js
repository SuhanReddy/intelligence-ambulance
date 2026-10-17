const modelMetrics = {
  SVM: { accuracy: 87.5, precision: 76.56, recall: 87.5, f1: 81.67 },
  'Random Forest': { accuracy: 89.58, precision: 88.06, recall: 89.58, f1: 88.03 },
  KNN: { accuracy: 87.5, precision: 76.56, recall: 87.5, f1: 81.67 }
};

const state = {
  dataset: [],
  features: [],
  target: 'target'
};

const datasetStatusEl = document.getElementById('dataset-status');
const featureFieldsEl = document.getElementById('feature-fields');
const modelTableBodyEl = document.getElementById('model-table-body');
const predictionForm = document.getElementById('prediction-form');
const datasetForm = document.getElementById('dataset-form');
const csvFileInput = document.getElementById('csv-file-input');
const resultsCard = document.getElementById('results-card');
const consensusEl = document.getElementById('consensus-label');
const confidenceEl = document.getElementById('confidence-value');
const voteEl = document.getElementById('vote-breakdown');

function renderModelTable() {
  modelTableBodyEl.innerHTML = '';

  Object.entries(modelMetrics).forEach(([name, metrics]) => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${name}</td>
      <td>${metrics.accuracy}%</td>
      <td>${metrics.precision}%</td>
      <td>${metrics.recall}%</td>
      <td>${metrics.f1}%</td>
    `;
    modelTableBodyEl.appendChild(row);
  });
}

function detectTarget(headers) {
  const candidates = ['target', 'label', 'class', 'condition', 'output'];
  const lower = headers.map((h) => h.trim().toLowerCase());

  for (const candidate of candidates) {
    const index = lower.indexOf(candidate);
    if (index !== -1) return headers[index];
  }

  return headers[headers.length - 1];
}

function parseCSV(text) {
  const rows = text.trim().split(/\r?\n/).filter((row) => row.trim() !== '');
  if (!rows.length) return { headers: [], data: [] };

  const headers = rows[0].split(',').map((h) => h.trim());
  const data = rows.slice(1).map((row) => {
    const values = row.split(',');
    const obj = {};
    headers.forEach((header, index) => {
      obj[header] = values[index] ? values[index].trim() : '';
    });
    return obj;
  });

  return { headers, data };
}

function populateFeatureFields() {
  featureFieldsEl.innerHTML = '';

  state.features.forEach((feature) => {
    const label = document.createElement('label');
    label.textContent = feature;

    const input = document.createElement('input');
    input.name = feature;
    input.type = 'number';
    input.step = 'any';
    input.required = true;

    input.value = '';
    input.placeholder = 'Enter value';
    label.appendChild(input);
    featureFieldsEl.appendChild(label);
  });
}

function computePrediction(formData) {
  const values = {};
  state.features.forEach((feature) => {
    values[feature] = Number(formData.get(feature));
  });

  const scored = state.dataset
    .map((row) => {
      let distance = 0;
      state.features.forEach((feature) => {
        const current = Number(row[feature]);
        const delta = current - values[feature];
        distance += delta * delta;
      });

      return {
        distance,
        classLabel: Number(row[state.target])
      };
    })
    .sort((a, b) => a.distance - b.distance);

  const neighbors = scored.slice(0, 5);
  const voteMap = {};
  neighbors.forEach(({ classLabel }) => {
    voteMap[classLabel] = (voteMap[classLabel] || 0) + 1;
  });

  const winner = Object.entries(voteMap).sort((a, b) => b[1] - a[1])[0];
  const label = winner ? Number(winner[0]) : 0;
  const nearest = neighbors[0]?.distance ?? 0;
  const confidence = Math.max(0, 100 - nearest * 18);

  return {
    label,
    confidence: Math.min(99.99, Math.max(40, confidence)).toFixed(2),
    voteBreakdown: Object.entries(voteMap)
      .sort((a, b) => b[1] - a[1])
      .map(([key, count]) => `${count}x class ${key}`)
      .join(' | ')
  };
}

function setDatasetFromText(csvText, sourceName = 'uploaded dataset') {
  const parsed = parseCSV(csvText);

  if (!parsed.headers.length || !parsed.data.length) {
    throw new Error('The CSV file is empty or malformed.');
  }

  state.target = detectTarget(parsed.headers);
  state.dataset = parsed.data;
  state.features = parsed.headers.filter((header) => header !== state.target);

  if (!state.features.length) {
    throw new Error('The CSV file must contain at least one feature column besides the target column.');
  }

  datasetStatusEl.textContent = `Loaded ${state.dataset.length} rows from ${sourceName}. Target: ${state.target}.`;
  populateFeatureFields();
}

async function loadDefaultDataset() {
  try {
    const response = await fetch('data/demo_heart_dataset.csv');
    if (!response.ok) throw new Error('Dataset not found');

    const csvText = await response.text();
    setDatasetFromText(csvText, 'the demo heart dataset');
    renderModelTable();
  } catch (error) {
    datasetStatusEl.textContent = 'Unable to load the demo dataset. Please upload a CSV file manually.';
    console.error(error);
  }
}

datasetForm.addEventListener('submit', (event) => {
  event.preventDefault();

  const file = csvFileInput.files[0];
  if (!file) {
    datasetStatusEl.textContent = 'Please choose a CSV file first.';
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    try {
      setDatasetFromText(String(reader.result), file.name);
      renderModelTable();
    } catch (error) {
      datasetStatusEl.textContent = error.message;
    }
  };
  reader.readAsText(file);
});

predictionForm.addEventListener('submit', (event) => {
  event.preventDefault();

  if (!state.dataset.length) {
    datasetStatusEl.textContent = 'Load a dataset before running prediction.';
    return;
  }

  const formData = new FormData(predictionForm);
  const prediction = computePrediction(formData);

  resultsCard.classList.remove('hidden');
  consensusEl.textContent = prediction.label === 1 ? 'High risk' : 'Lower risk';
  confidenceEl.textContent = `${prediction.confidence}%`;
  voteEl.textContent = prediction.voteBreakdown;
});

loadDefaultDataset();

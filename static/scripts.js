document.getElementById('startBtn').addEventListener('click', () => {
  document.querySelector('.intro').style.display = 'none';
  document.querySelector('form').style.display = 'block';
  showStep(0);
});

let currentStep = 0;
const steps = document.querySelectorAll('.step');

function showStep(index) {
  steps.forEach((s, i) => s.style.display = i === index ? 'block' : 'none');
  if (index < steps.length - 1) {
    steps[index].addEventListener('change', () => showStep(index + 1), { once: true });
  }
}
const API_URL = 'http://localhost:8000/api/v1';
let currentToken = localStorage.getItem('token');
let user = null;
let currentQuestionIndex = 0;
let answers = {
    R: 0, I: 0, A: 0, S: 0, E: 0, C: 0
};
let authMode = 'register'; // register or login

const questions = [
    // Realistic
    { text: "¿Te gustaría trabajar reparando aparatos electrónicos o motores?", cat: 'R' },
    { text: "¿Te interesa el manejo de herramientas y maquinaria pesada?", cat: 'R' },
    { text: "¿Te agrada realizar actividades físicas al aire libre?", cat: 'R' },
    { text: "¿Te gustaría aprender cómo se construyen puentes o edificios?", cat: 'R' },
    { text: "¿Te sientes cómodo realizando trabajos manuales de precisión?", cat: 'R' },
    // Investigative
    { text: "¿Te apasiona investigar el porqué de los fenómenos naturales?", cat: 'I' },
    { text: "¿Disfrutas resolviendo problemas matemáticos complejos?", cat: 'I' },
    { text: "¿Te gustaría trabajar en un laboratorio haciendo experimentos?", cat: 'I' },
    { text: "¿Te interesa leer sobre avances científicos y tecnología?", cat: 'I' },
    { text: "¿Te gusta analizar datos para encontrar patrones?", cat: 'I' },
    // Artistic
    { text: "¿Te gusta expresar tus ideas a través del dibujo o la pintura?", cat: 'A' },
    { text: "¿Te gustaría escribir cuentos, poemas o artículos de opinión?", cat: 'A' },
    { text: "¿Disfrutas tocando un instrumento musical o componiendo?", cat: 'A' },
    { text: "¿Te interesa el diseño de modas o la decoración de interiores?", cat: 'A' },
    { text: "¿Te gustaría actuar en una obra de teatro o película?", cat: 'A' },
    // Social
    { text: "¿Te sientes bien ayudando a personas con problemas personales?", cat: 'S' },
    { text: "¿Te gustaría trabajar como profesor o instructor?", cat: 'S' },
    { text: "¿Te interesa el cuidado de la salud y el bienestar de los demás?", cat: 'S' },
    { text: "¿Disfrutas participar en voluntariados o proyectos sociales?", cat: 'S' },
    { text: "¿Te gusta escuchar y dar consejos a tus amigos?", cat: 'S' },
    // Enterprising
    { text: "¿Te gustaría dirigir tu propia empresa o negocio?", cat: 'E' },
    { text: "¿Te sientes cómodo hablando en público para convencer a otros?", cat: 'E' },
    { text: "¿Te interesa el mundo de las ventas y el marketing?", cat: 'E' },
    { text: "¿Te gustaría liderar equipos de trabajo en grandes proyectos?", cat: 'E' },
    { text: "¿Te atrae la idea de ser un abogado o negociador?", cat: 'E' },
    // Conventional
    { text: "¿Te gusta mantener tus cosas perfectamente ordenadas y clasificadas?", cat: 'C' },
    { text: "¿Te gustaría trabajar con hojas de cálculo y presupuestos?", cat: 'C' },
    { text: "¿Te interesa asegurar que se cumplan las normas y reglamentos?", cat: 'C' },
    { text: "¿Te sientes cómodo realizando tareas administrativas de oficina?", cat: 'C' },
    { text: "¿Te gusta llevar un registro detallado de gastos e ingresos?", cat: 'C' }
];

// Initialize
if (currentToken) {
    checkUser();
}

function showAuth(mode) {
    authMode = mode;
    document.getElementById('hero').style.display = 'none';
    document.getElementById('auth-section').style.display = 'block';
    updateAuthUI();
}

function toggleAuthMode() {
    authMode = (authMode === 'register' ? 'login' : 'register');
    updateAuthUI();
}

function updateAuthUI() {
    const title = document.getElementById('auth-title');
    const nameGroup = document.getElementById('name-group');
    const toggleText = document.getElementById('auth-toggle-text');
    const toggleLink = document.getElementById('auth-toggle-link');
    
    if (authMode === 'login') {
        title.innerText = 'Iniciar Sesión';
        nameGroup.style.display = 'none';
        toggleText.innerText = '¿No tienes cuenta?';
        toggleLink.innerText = 'Regístrate';
    } else {
        title.innerText = 'Registrarse';
        nameGroup.style.display = 'block';
        toggleText.innerText = '¿Ya tienes cuenta?';
        toggleLink.innerText = 'Inicia sesión';
    }
}

async function handleAuth(e) {
    e.preventDefault();
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const fullname = document.getElementById('fullname').value;

    try {
        if (authMode === 'register') {
            await fetch(`${API_URL}/auth/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password, full_name: fullname })
            });
            authMode = 'login';
            handleAuth(e); // Auto login after register
            return;
        }

        // Login
        const formData = new FormData();
        formData.append('username', email);
        formData.append('password', password);

        const response = await fetch(`${API_URL}/auth/login`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Error en autenticación');
        
        const data = await response.json();
        currentToken = data.access_token;
        localStorage.setItem('token', currentToken);
        
        document.getElementById('auth-section').style.display = 'none';
        document.getElementById('hero').style.display = 'flex';
        checkUser();
    } catch (err) {
        alert(err.message);
    }
}

async function checkUser() {
    // In a real app, we'd fetch user info here.
    document.getElementById('auth-links').style.display = 'none';
    document.getElementById('user-info').style.display = 'flex';
}

function logout() {
    localStorage.removeItem('token');
    location.reload();
}

function startTest() {
    if (!currentToken) {
        showAuth('register');
        return;
    }
    document.getElementById('hero').style.display = 'none';
    document.getElementById('survey').style.display = 'block';
    showQuestion();
}

function showQuestion() {
    const q = questions[currentQuestionIndex];
    document.getElementById('question-text').innerText = q.text;
    document.getElementById('current-q').innerText = currentQuestionIndex + 1;
    
    const progress = ((currentQuestionIndex + 1) / questions.length) * 100;
    document.getElementById('progress-fill').style.width = `${progress}%`;
}

function answerQuestion(value) {
    const q = questions[currentQuestionIndex];
    answers[q.cat] += value;
    
    if (currentQuestionIndex < questions.length - 1) {
        currentQuestionIndex++;
        showQuestion();
    } else {
        submitTest();
    }
}

async function submitTest() {
    document.getElementById('survey').innerHTML = '<div style="text-align:center; padding: 50px;"><i class="fas fa-spinner fa-spin fa-3x"></i><p style="margin-top:20px;">Procesando con Árbol de Decisión...</p></div>';
    
    try {
        const response = await fetch(`${API_URL}/test/submit`, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${currentToken}`
            },
            body: JSON.stringify({ scores: answers })
        });

        if (!response.ok) throw new Error('Error al procesar el test');
        
        const result = await response.json();
        showResults(result);
    } catch (err) {
        alert(err.message);
        location.reload();
    }
}

function showResults(data) {
    document.getElementById('survey').style.display = 'none';
    document.getElementById('results').style.display = 'block';
    
    document.getElementById('res-category').innerText = data.recommendation;
    
    // Split recommendation to get description and careers
    const parts = data.details.split(' Carreras sugeridas: ');
    document.getElementById('res-description').innerText = parts[0];
    
    const careers = parts[1].split(', ');
    const careersGrid = document.getElementById('res-careers');
    careersGrid.innerHTML = '';
    
    careers.forEach(career => {
        const card = document.createElement('div');
        card.className = 'career-card glass';
        card.innerHTML = `
            <h3>${career}</h3>
            <p>Carrera profesional de alta demanda en el mercado peruano.</p>
        `;
        careersGrid.appendChild(card);
    });
}

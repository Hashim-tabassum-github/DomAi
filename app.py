# app.py

from flask import Flask, render_template, request, redirect, url_for, flash, session, g
import os
from werkzeug.utils import secure_filename
from tran import MultiTaskNet, predict, OfficeHomeDataset, DEVICE
import torch
import torchvision.transforms as transforms
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# --- Setup ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'officehome_multitask_model.pth'
DATASET_PATH = r'E:\data\office_home'
DATABASE = os.path.join(os.path.dirname(__file__), 'users.db')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
full_dataset = OfficeHomeDataset(root_dir=DATASET_PATH, transform=None)
num_classes = len(full_dataset.classes)
num_domains = len(full_dataset.domains)
model = MultiTaskNet(num_classes=num_classes, num_domains=num_domains)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Database Functions ---
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''CREATE TABLE IF NOT EXISTS user (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            profile_image TEXT
        )''')
        try:
            db.execute('ALTER TABLE user ADD COLUMN profile_image TEXT')
            db.commit()
        except sqlite3.OperationalError:
            pass

init_db()

# --- Decorators ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('is_admin'):
            flash('You must be an admin to view this page.', 'warning')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# --- Routes ---
@app.route('/')
def root():
    if 'user_id' in session:
        if session.get('is_admin'):
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('root'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # --- Admin Login Check ---
        if email == 'admin@domai.ai' and password == '123123':
            session.clear()
            session['user_id'] = 'admin'
            session['user_name'] = 'Administrator'
            session['is_admin'] = True
            # <<< MODIFIED: Set a default profile image for the admin >>>
            session['profile_image'] = 'admin_profile.png'
            flash('Admin login successful!', 'success')
            return redirect(url_for('admin_dashboard'))

        # --- Regular User Login ---
        db = get_db()
        user = db.execute('SELECT * FROM user WHERE email = ?', (email,)).fetchone()
        
        if user and check_password_hash(user['password_hash'], password):
            session.clear()
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            session['profile_image'] = user['profile_image'] or 'default_profile.png'
            session['is_admin'] = False
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'danger')
            
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm = request.form.get('confirm')
        if not name or not email or not password or not confirm:
            flash('Please fill all fields.', 'danger')
        elif password != confirm:
            flash('Passwords do not match.', 'danger')
        else:
            db = get_db()
            try:
                db.execute('INSERT INTO user (name, email, password_hash, profile_image) VALUES (?, ?, ?, ?)',
                           (name, email, generate_password_hash(password), 'default_profile.png'))
                db.commit()
                user = db.execute('SELECT * FROM user WHERE email = ?', (email,)).fetchone()
                session['user_id'] = user['id']
                session['user_name'] = user['name']
                session['profile_image'] = user['profile_image'] or 'default_profile.png'
                return redirect(url_for('dashboard'))
            except sqlite3.IntegrityError:
                flash('Email already registered.', 'danger')
    return render_template('signup.html')


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result = predict(
                model=model,
                image_path=filepath,
                transform=inference_transform,
                device=DEVICE,
                idx_to_class=full_dataset.idx_to_class,
                idx_to_domain=full_dataset.idx_to_domain
            )
            return render_template('result.html', result=result, image_url=url_for('uploaded_file', filename=filename))
        else:
            flash('Allowed file types are png, jpg, jpeg')
            return redirect(request.url)
    return render_template('index.html')


@app.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    db = get_db()
    users = db.execute('SELECT id, name, email, profile_image FROM user ORDER BY id').fetchall()
    return render_template('admin.html', users=users)


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    db = get_db()

    # <<< NEW LOGIC: Handle admin and regular users differently >>>
    if session.get('is_admin'):
        # For the admin, create a dictionary from session data (no DB call)
        user_dict = {
            'id': session['user_id'],
            'name': session['user_name'],
            'email': 'admin@domai.ai',
            'profile_image': session.get('profile_image', 'default_profile.png')
        }
        # Prevent the admin from changing their profile picture
        if request.method == 'POST':
            flash('Admin profile cannot be modified.', 'warning')
    else:
        # For regular users, fetch data from the database
        user = db.execute('SELECT id, name, email, profile_image FROM user WHERE id = ?', (session['user_id'],)).fetchone()
        
        if not user:
            flash('User not found.', 'danger')
            return redirect(url_for('dashboard'))
            
        user_dict = {
            'id': user['id'], 
            'name': user['name'], 
            'email': user['email'], 
            'profile_image': user['profile_image'] or 'default_profile.png'
        }

        if request.method == 'POST':
            file = request.files.get('profile_image')
            if file and file.filename:
                filename = f"profile_{session['user_id']}.png"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                db.execute('UPDATE user SET profile_image = ? WHERE id = ?', (filename, session['user_id']))
                db.commit()
                
                session['profile_image'] = filename
                user_dict['profile_image'] = filename  # Update dict to show new image immediately
                flash('Profile image updated!', 'success')

    return render_template('profile.html', user=user_dict)


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(debug=True)
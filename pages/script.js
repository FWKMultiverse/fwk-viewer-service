/**
 * =================================================================
 * FWK Viewer - Authentication & Content Protection (V3)
 * (แก้ไข: ดึงโค้ด Python โดยตรงจากไฟล์ .py ในโฟลเดอร์เดียวกัน)
 * =================================================================
 */

// --- 0. Import โค้ด Python ดิบจากไฟล์ .py ในโฟลเดอร์เดียวกัน (ใช้ ?raw ของ Vite)
// Path ถูกต้อง: ./filename.py?raw
import V9_CODE_RAW from './multiverse_v9.py?raw'; 
import PROTOTYPE_CODE_RAW from './model.py?raw'; 

// --- 1. การตั้งค่าคงที่ (Constants) ---------------------------------
const Config = (() => {
    const rawKeys = import.meta.env.VITE_VALID_KEYS || '';
    const adminKey = import.meta.env.VITE_ADMIN_KEY || 'DEFAULT-ADMIN-KEY'; 
    const expirationDays = parseInt(import.meta.env.VITE_EXPIRATION_DAYS, 10) || 30;

    return {
        VALID_KEYS: rawKeys.split(',').filter(k => k.trim() !== ''),
        ADMIN_KEY: adminKey.trim(),
        EXPIRATION_DAYS: expirationDays,
        STORAGE_KEY_NAME: 'fwk_active_key_v3',
        STORAGE_KEY_TIME: 'fwk_expiration_time_v3'
    };
})();
// =================================================================

// --- 2. Local Storage Helper Functions -----------------------------
const StorageHelper = {
    getActiveKey: () => localStorage.getItem(Config.STORAGE_KEY_NAME),
    getExpirationTime: () => localStorage.getItem(Config.STORAGE_KEY_TIME),
    setAccess: (key, expirationTime) => {
        localStorage.setItem(Config.STORAGE_KEY_NAME, key);
        localStorage.setItem(Config.STORAGE_KEY_TIME, expirationTime.toString());
    },
    clearAccess: () => {
        localStorage.removeItem(Config.STORAGE_KEY_NAME);
        localStorage.removeItem(Config.STORAGE_KEY_TIME);
    }
};

// --- 3. การจัดการ DOM และ State ------------------------------------
document.addEventListener('DOMContentLoaded', () => {

    // --- 3.1 การเลือกองค์ประกอบ DOM
    const DOM = {
        loginContainer: document.getElementById('login-container'),
        keyInput: document.getElementById('key-input'),
        loginButton: document.getElementById('login-button'),
        errorMessage: document.getElementById('error-message'),
        viewerContainer: document.getElementById('viewer-container'),
        timerDisplay: document.getElementById('timer-display'),
        btnPrototype: document.getElementById('btn-prototype'),
        btnV9: document.getElementById('btn-v9'),
        wrapperPrototype: document.getElementById('wrapper-prototype'),
        wrapperV9: document.getElementById('wrapper-v9'),
        codePrototype: document.getElementById('code-prototype'),
        codeV9: document.getElementById('code-v9')
    };

    let timerInterval = null;

    // --- 3.2 ฟังก์ชันแสดง/ซ่อน UI
    function showLogin(message = "") {
        if (timerInterval) clearInterval(timerInterval);
        DOM.loginContainer.style.display = 'flex';
        DOM.viewerContainer.style.display = 'none';
        DOM.errorMessage.textContent = message;
        DOM.keyInput.value = "";
    }

    function showViewer(expirationTime, isAdmin = false) {
        DOM.loginContainer.style.display = 'none';
        DOM.viewerContainer.style.display = 'block';

        loadAllCodeContent(); // โหลดโค้ด Python เมื่อเข้าสู่ Viewer

        if (isAdmin) {
            DOM.timerDisplay.textContent = "สถานะ: ADMIN (ไม่หมดอายุ)";
            if (timerInterval) clearInterval(timerInterval);
        } else {
            updateTimer(expirationTime);
            if (timerInterval) clearInterval(timerInterval);
            timerInterval = setInterval(() => updateTimer(expirationTime), 1000);
        }

        showTab('v9');
    }

    // --- 3.3 ฟังก์ชันการจัดการเวลาและแท็บ
    function updateTimer(expirationTime) {
        const now = Date.now();
        const remainingMs = expirationTime - now;

        if (remainingMs <= 0) {
            DOM.timerDisplay.textContent = "หมดอายุ";
            checkAccess();
            return;
        }

        const days = Math.floor(remainingMs / (1000 * 60 * 60 * 24));
        const hours = Math.floor((remainingMs % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
        const minutes = Math.floor((remainingMs % (1000 * 60 * 60)) / (1000 * 60));
        const seconds = Math.floor((remainingMs % (1000 * 60)) / 1000);

        DOM.timerDisplay.textContent = `เหลือเวลา: ${days} วัน ${hours} ชม. ${minutes} นาที ${seconds} วิ`;
    }

    function showTab(tabToShow) {
        const isV9 = tabToShow === 'v9';
        DOM.wrapperV9.style.display = isV9 ? 'block' : 'none';
        DOM.wrapperPrototype.style.display = isV9 ? 'none' : 'block';

        DOM.btnV9.classList.toggle('active', isV9);
        DOM.btnPrototype.classList.toggle('active', !isV9);
    }

    // *** ฟังก์ชันที่แก้ไข: โหลดโค้ด Python โดยใช้ String จาก Import ***
    function loadAllCodeContent() {
        // โหลด V9 Code
        if (DOM.codeV9) {
            DOM.codeV9.textContent = V9_CODE_RAW;
            if (typeof Prism !== 'undefined') {
                Prism.highlightElement(DOM.codeV9);
            }
        }
        
        // โหลด Prototype Code
        if (DOM.codePrototype) {
            DOM.codePrototype.textContent = PROTOTYPE_CODE_RAW;
            if (typeof Prism !== 'undefined') {
                Prism.highlightElement(DOM.codePrototype);
            }
        }
    }


    // --- 3.4 ฟังก์ชันการตรวจสอบและการเข้าสู่ระบบ

    function checkAccess() {
        const activeKey = StorageHelper.getActiveKey();
        const expirationTimeStr = StorageHelper.getExpirationTime();

        // ตรวจสอบ: หาก Config.VALID_KEYS ไม่มีเลย แสดงว่าการดึงค่าล้มเหลว
        if (Config.VALID_KEYS.length === 0 && activeKey !== Config.ADMIN_KEY) {
             console.error("CONFIGURATION ERROR: Valid keys not loaded. Check ENV setup.");
             showLogin("ข้อผิดพลาด: ระบบคีย์มีปัญหา โปรดติดต่อผู้ดูแล");
             return;
        }

        if (!activeKey || !expirationTimeStr) {
            showLogin();
            return;
        }

        const now = Date.now();
        const expirationTime = parseInt(expirationTimeStr, 10);

        if (now > expirationTime) {
            StorageHelper.clearAccess();
            showLogin("คีย์ของคุณหมดอายุแล้ว");
            return;
        }

        showViewer(expirationTime, activeKey === Config.ADMIN_KEY); 
    }

    function handleLogin() {
        const key = DOM.keyInput.value.trim();

        if (key === Config.ADMIN_KEY) {
            showViewer(Date.now() + (365 * 24 * 60 * 60 * 1000), true);
            return;
        }

        if (Config.VALID_KEYS.includes(key)) {
            const expirationTime = Date.now() + (Config.EXPIRATION_DAYS * 24 * 60 * 60 * 1000);

            StorageHelper.setAccess(key, expirationTime);

            showViewer(expirationTime, false);
        } else {
            showLogin("คีย์ไม่ถูกต้อง หรือ ไม่พบสิทธิ์นี้");
        }
    }

    // --- 3.5 การกำหนด Event Listeners
    function setupEventListeners() {
        DOM.loginButton.addEventListener('click', handleLogin);
        DOM.keyInput.addEventListener('keyup', (e) => {
            if (e.key === 'Enter') handleLogin();
        });
        DOM.btnPrototype.addEventListener('click', () => showTab('prototype'));
        DOM.btnV9.addEventListener('click', () => showTab('v9'));
    }

    // --- 3.6 เริ่มการทำงาน
    setupEventListeners();
    checkAccess();
});

// =================================================================
//  4. มาตรการความปลอดภัยพื้นฐาน (ป้องกันการคัดลอก/ตรวจสอบ)
// =================================================================
(function protectContent() {
    // 4.1 ป้องกันการคลิกขวา, เลือกข้อความ, และลากเนื้อหา
    document.addEventListener('contextmenu', e => e.preventDefault());
    document.addEventListener('selectstart', e => e.preventDefault());
    document.addEventListener('dragstart', e => e.preventDefault());

    // 4.2 ป้องกันปุ่มลัดสำคัญ (Ctrl+C, F12, Ctrl+Shift+I)
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey && (e.key === 'c' || e.key === 'a' || e.key === 's' || e.key === 'p')) {
            e.preventDefault();
        }
        if (e.key === 'F12' || (e.ctrlKey && e.shiftKey && (e.key === 'I' || e.key === 'i' || e.key === 'J' || e.key === 'j'))) {
            e.preventDefault();
        }
    });

    // 4.3 ป้องกัน PrintScreen (บางเบราว์เซอร์/ระบบปฏิบัติการ)
    document.addEventListener('keyup', (e) => {
        if (e.key === 'PrintScreen' || e.code === 'PrintScreen') {
            alert("ระบบป้องกัน: ไม่อนุญาตให้จับภาพหน้าจอ");
        }
    });

    // 4.4 ตรวจจับ Developer Tools
    let devtoolsOpen = false;
    const checkDevTools = () => {
        const diffHeight = window.outerHeight - window.innerHeight;
        const diffWidth = window.outerWidth - window.innerWidth;

        if (diffHeight > 150 || diffWidth > 150) {
            if (!devtoolsOpen) {
                document.body.innerHTML = "<h3 style='color:red;text-align:center;margin-top:30vh;'>ปิด Developer Tools ก่อนใช้งาน</h3>";
                devtoolsOpen = true;
            }
        } else if (devtoolsOpen) {
             window.location.reload(); 
        }
    };
    setInterval(checkDevTools, 1000);


    // 4.5 วาง overlay โปร่งใส
    (() => {
        const overlay = document.createElement('div');
        Object.assign(overlay.style, {
            position: 'fixed', top: 0, left: 0,
            width: '100vw', height: '100vh', zIndex: 9999,
            background: 'rgba(255,255,255,0)',
            pointerEvents: 'none'
        });
        document.body.appendChild(overlay);
    })();
})();
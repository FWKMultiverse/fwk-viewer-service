// vite.config.js
import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
    // 1. กำหนด Base URL เป็นชื่อ Repository (สำหรับ Path ภายใน JS/CSS)
    base: '/fwk-viewer-service/', 
    
    build: {
        // 2. *** ส่วนสำคัญ: เปลี่ยนการตั้งค่า Rollup เพื่อบอกว่า index.html อยู่ที่ Root ***
        rollupOptions: {
            input: {
                // ชี้ Vite ไปที่ไฟล์ index.html ในโฟลเดอร์ pages/
                // แต่ Rollup จะวางมันไว้ที่ Root ของ dist/
                main: resolve(__dirname, 'pages/index.html') 
            }
        }
        // ไม่ต้องระบุ outDir เพราะต้องการให้ใช้ค่าเริ่มต้น (dist/)
    },
});
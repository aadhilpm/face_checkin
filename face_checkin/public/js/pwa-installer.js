// PWA Installation Prompt and Management
class PWAInstaller {
    constructor() {
        this.deferredPrompt = null;
        this.installButton = null;
        this.init();
    }

    init() {
        // Listen for beforeinstallprompt event
        window.addEventListener('beforeinstallprompt', (e) => {
            // Prevent the mini-infobar from appearing on mobile
            e.preventDefault();
            // Store the event so it can be triggered later
            this.deferredPrompt = e;
            // Show install button
            this.showInstallButton();
        });

        // Listen for app installed event
        window.addEventListener('appinstalled', () => {
            console.log('PWA was installed');
            this.hideInstallButton();
            this.showInstalledMessage();
        });

        // Check if app is already installed
        if (this.isAppInstalled()) {
            this.hideInstallButton();
        }

        // Create install button
        this.createInstallButton();

        // Register service worker if not already registered
        this.registerServiceWorker();
    }

    createInstallButton() {
        // Create install button element
        this.installButton = document.createElement('button');
        this.installButton.id = 'pwa-install-btn';
        this.installButton.innerHTML = `
            <i class="fas fa-download"></i>
            Install App
        `;
        this.installButton.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: linear-gradient(135deg, #3b82f6, #1d4ed8);
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 50px;
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
            z-index: 1000;
            display: none;
            align-items: center;
            gap: 8px;
            transition: all 0.3s ease;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        `;

        // Add hover effect
        this.installButton.addEventListener('mouseenter', () => {
            this.installButton.style.transform = 'translateY(-2px)';
            this.installButton.style.boxShadow = '0 8px 25px rgba(59, 130, 246, 0.5)';
        });

        this.installButton.addEventListener('mouseleave', () => {
            this.installButton.style.transform = 'translateY(0)';
            this.installButton.style.boxShadow = '0 4px 20px rgba(59, 130, 246, 0.4)';
        });

        // Add click handler
        this.installButton.addEventListener('click', () => {
            this.promptInstall();
        });

        // Add to page
        document.body.appendChild(this.installButton);
    }

    showInstallButton() {
        if (this.installButton && !this.isAppInstalled()) {
            this.installButton.style.display = 'flex';
            
            // Animate in
            setTimeout(() => {
                this.installButton.style.opacity = '1';
                this.installButton.style.transform = 'translateY(0)';
            }, 100);
        }
    }

    hideInstallButton() {
        if (this.installButton) {
            this.installButton.style.display = 'none';
        }
    }

    async promptInstall() {
        if (!this.deferredPrompt) {
            // Fallback for browsers that don't support beforeinstallprompt
            this.showManualInstallInstructions();
            return;
        }

        // Show the install prompt
        this.deferredPrompt.prompt();

        // Wait for the user to respond to the prompt
        const { outcome } = await this.deferredPrompt.userChoice;
        
        if (outcome === 'accepted') {
            console.log('User accepted the install prompt');
            this.hideInstallButton();
        } else {
            console.log('User dismissed the install prompt');
        }

        // Clear the saved prompt since it can only be used once
        this.deferredPrompt = null;
    }

    showManualInstallInstructions() {
        const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent);
        const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
        
        let instructions = '';
        
        if (isIOS && isSafari) {
            instructions = `
                <h3>Install Face Check-in App</h3>
                <p>To install this app on your iOS device:</p>
                <ol>
                    <li>Tap the Share button <i class="fas fa-share"></i></li>
                    <li>Scroll down and tap "Add to Home Screen"</li>
                    <li>Tap "Add" to confirm</li>
                </ol>
            `;
        } else {
            instructions = `
                <h3>Install Face Check-in App</h3>
                <p>To install this app:</p>
                <ol>
                    <li>Click the menu (⋮) in your browser</li>
                    <li>Look for "Install app" or "Add to Home Screen"</li>
                    <li>Follow the prompts to install</li>
                </ol>
            `;
        }

        this.showModal(instructions);
    }

    showInstalledMessage() {
        this.showModal(`
            <h3><i class="fas fa-check-circle" style="color: #10b981;"></i> App Installed!</h3>
            <p>Face Check-in has been installed successfully. You can now access it from your home screen or app drawer.</p>
        `);
    }

    showModal(content) {
        // Create modal
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
            backdrop-filter: blur(5px);
        `;

        const modalContent = document.createElement('div');
        modalContent.style.cssText = `
            background: rgba(31, 41, 55, 0.95);
            border: 1px solid rgba(75, 85, 99, 0.3);
            border-radius: 16px;
            padding: 2rem;
            max-width: 400px;
            width: 90%;
            color: #f9fafb;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        `;

        modalContent.innerHTML = `
            ${content}
            <button onclick="this.closest('.pwa-modal').remove()" style="
                background: linear-gradient(135deg, #3b82f6, #1d4ed8);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-weight: 600;
                cursor: pointer;
                margin-top: 1rem;
                width: 100%;
            ">Close</button>
        `;

        modal.className = 'pwa-modal';
        modal.appendChild(modalContent);
        document.body.appendChild(modal);

        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (modal.parentNode) {
                modal.remove();
            }
        }, 10000);
    }

    isAppInstalled() {
        // Check if app is installed (running in standalone mode)
        return window.matchMedia('(display-mode: standalone)').matches ||
               window.navigator.standalone ||
               document.referrer.includes('android-app://');
    }

    async registerServiceWorker() {
        if ('serviceWorker' in navigator) {
            try {
                const registration = await navigator.serviceWorker.register('/assets/face_checkin/js/sw.js');
                console.log('Service Worker registered:', registration);
                
                // Check for updates
                registration.addEventListener('updatefound', () => {
                    const newWorker = registration.installing;
                    newWorker.addEventListener('statechange', () => {
                        if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                            this.showUpdateAvailable(registration);
                        }
                    });
                });
            } catch (error) {
                console.error('Service Worker registration failed:', error);
            }
        }
    }

    showUpdateAvailable(registration) {
        const updateButton = document.createElement('button');
        updateButton.innerHTML = `
            <i class="fas fa-sync-alt"></i>
            Update Available
        `;
        updateButton.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 50px;
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(16, 185, 129, 0.4);
            z-index: 1000;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.3s ease;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        `;

        updateButton.addEventListener('click', () => {
            if (registration.waiting) {
                registration.waiting.postMessage({ type: 'SKIP_WAITING' });
                window.location.reload();
            }
        });

        document.body.appendChild(updateButton);

        // Auto-remove after 30 seconds
        setTimeout(() => {
            if (updateButton.parentNode) {
                updateButton.remove();
            }
        }, 30000);
    }
}

// Initialize PWA installer when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new PWAInstaller();
});

// Add some PWA utility functions to global scope
window.PWAUtils = {
    isInstalled: () => {
        return window.matchMedia('(display-mode: standalone)').matches ||
               window.navigator.standalone ||
               document.referrer.includes('android-app://');
    },
    
    canInstall: () => {
        return 'serviceWorker' in navigator && 'PushManager' in window;
    },
    
    getInstallStatus: () => {
        return {
            canInstall: window.PWAUtils.canInstall(),
            isInstalled: window.PWAUtils.isInstalled(),
            isOnline: navigator.onLine,
            serviceWorkerSupported: 'serviceWorker' in navigator
        };
    }
};
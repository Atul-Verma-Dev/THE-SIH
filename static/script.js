// Wait for DOM to load
document.addEventListener('DOMContentLoaded', () => {
    
    // 1. Mobile Navbar Hamburger Menu
    const hamburger = document.getElementById('hamburger');
    const navLinks = document.getElementById('nav-links');

    if(hamburger && navLinks) {
        hamburger.addEventListener('click', () => {
            // Toggle active class on navigation links wrapper
            navLinks.classList.toggle('active');
            
            // Animate Hamburger (optional enhancement)
            hamburger.classList.toggle('toggle');
        });
    }

    // Close mobile menu when a link is clicked
    const links = document.querySelectorAll('.nav-links li a');
    links.forEach(link => {
        link.addEventListener('click', () => {
            if(navLinks.classList.contains('active')) {
                navLinks.classList.remove('active');
            }
        });
    });

    // 2. Contact Form Demonstration Handler
    const demoContactForm = document.getElementById('demo-contact-form');
    const formAlert = document.getElementById('form-alert');

    if(demoContactForm && formAlert) {
        demoContactForm.addEventListener('submit', (e) => {
            // Prevent actual form submission (No backend)
            e.preventDefault();
            
            // Show the prototype demo message
            formAlert.classList.remove('hidden');
            
            // Reset the form fields
            demoContactForm.reset();
            
            // Hide the message after 5 seconds
            setTimeout(() => {
                formAlert.classList.add('hidden');
            }, 5000);
        });
    }
});
// Language translations for index.html
const translations = {
  hindi: {
    h1: "ऑनबोर्डिंग और भाषा चयन",
    hindiBtn: "हिंदी",
    englishBtn: "अंग्रेज़ी",
    end: "यह पूरे ऐप अनुभव के लिए भाषा सेट करता है।"
  },
  english: {
    h1: "Onboarding & Language Selection",
    hindiBtn: "हिंदी",
    englishBtn: "English",
    end: "This sets the language for the entire app experience."
  }
};

function setLanguage(lang) {
  localStorage.setItem('language', lang);
  applyLanguage();
}

function applyLanguage() {
  const lang = localStorage.getItem('language') || 'english';
  const t = translations[lang];
  document.querySelector('h1').innerText = t.h1;
  document.querySelector('.buttons a:nth-child(1)').innerText = t.hindiBtn;
  document.querySelector('.buttons a:nth-child(2)').innerText = t.englishBtn;
  document.querySelector('.end').innerText = t.end;
}

document.addEventListener('DOMContentLoaded', function() {
  applyLanguage();
  document.querySelector('.buttons a:nth-child(1)').onclick = function(e) {
    e.preventDefault();
    setLanguage('hindi');
  };
  document.querySelector('.buttons a:nth-child(2)').onclick = function(e) {
    e.preventDefault();
    setLanguage('english');
  };
});

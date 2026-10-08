"""
Tresvance knowledge base — content sourced from https://www.tresvance.com/
(home, /our-works, /contact). Each entry is one retrievable chunk for RAG.
Update this file whenever the live site content changes.
"""

DOCUMENTS = [
    {
        "id": "about-story",
        "text": (
            "About Tresvance: Tresvance is a software, AI, IoT and cybersecurity company "
            "based in Kochi (Cochin), Kerala, India. The company was founded on the idea that "
            "technology should not lose its human element — they describe themselves as a "
            "collective that thinks strategically but builds with the passion of artists, "
            "bridging the gap between raw engineering and genuine human connection. "
            "Tresvance has completed 25 projects, has 5+ years of experience, and has served "
            "20 happy clients."
        ),
    },
    {
        "id": "services-overview",
        "text": (
            "Tresvance offers six core services: (1) Full-Stack Software Engineering, "
            "(2) Next-Gen AI Development, (3) IoT Device Implementation, "
            "(4) End-to-End Cybersecurity, (5) Results-Driven Digital Marketing, and "
            "(6) IT, Managed Services & Digital Strategy."
        ),
    },
    {
        "id": "service-software-engineering",
        "text": (
            "Service: Full-Stack Software Engineering. Tresvance designs and builds robust, "
            "scalable, tailor-made digital products — from intuitive web apps to complex "
            "enterprise systems — engineered to fit each client's unique business workflows."
        ),
    },
    {
        "id": "service-ai-development",
        "text": (
            "Service: Next-Gen AI Development. Tresvance accelerates business efficiency by "
            "building and integrating smart AI models, automating workflows, and leveraging "
            "machine learning to solve complex operational challenges."
        ),
    },
    {
        "id": "service-iot",
        "text": (
            "Service: IoT Device Implementation. Tresvance helps businesses transform through "
            "IoT integration, device management, and data-driven automation."
        ),
    },
    {
        "id": "service-cybersecurity",
        "text": (
            "Service: End-to-End Cybersecurity. Tresvance protects digital assets with "
            "comprehensive security services, including threat detection, risk assessments, "
            "and robust defense protocols, built on zero-trust architecture."
        ),
    },
    {
        "id": "service-digital-marketing",
        "text": (
            "Service: Results-Driven Digital Marketing. Tresvance scales online presence and "
            "accelerates brand growth through targeted marketing strategies, SEO, social media "
            "management, and data-backed campaigns."
        ),
    },
    {
        "id": "service-managed-it",
        "text": (
            "Service: IT, Managed Services & Digital Strategy. Tresvance provides proactive IT "
            "management, cloud infrastructure support, and continuous network monitoring, "
            "alongside strategic digital transformation to modernize legacy systems and "
            "optimize operations."
        ),
    },
    {
        "id": "product-tres-ai",
        "text": (
            "Product: TRES AI Assistant. This is Tresvance's AI business assistant product — "
            "it answers customers instantly, qualifies leads, books appointments, and sends "
            "automatic follow-ups, 24/7, across a business's website, WhatsApp, and social "
            "media inboxes. Tagline: 'from Hi to booked, 24/7.' Key features: 24/7 replies, "
            "lead qualification, smart booking, and auto follow-ups."
        ),
    },
    {
        "id": "why-choose-us",
        "text": (
            "Why choose Tresvance: (1) Cutting-Edge Expertise — the team builds advanced "
            "machine learning models and enterprise full-stack ecosystems with deep technical "
            "mastery. (2) Precision Security Frameworks — every product is hardened against "
            "vulnerabilities from day one using zero-trust architecture. (3) Client-Centric "
            "Architecture — clients get complete code ownership, transparent documentation, "
            "agile delivery, and solutions built around their specific business needs. "
            "(4) Scalable Data Engines — AI workflows and IoT integrations automate business "
            "operations and turn raw data into structured decisions."
        ),
    },
    {
        "id": "project-bmce",
        "text": (
            "Project: BMCE College Website. A comprehensive web application for BMCE College "
            "to streamline academic operations and enhance digital accessibility, built for "
            "performance, scalability, and user-friendly navigation for a modern educational "
            "experience."
        ),
    },
    {
        "id": "project-biohippo",
        "text": (
            "Project: BioHippo. An innovative web platform for scientists and researchers to "
            "source, compare, and purchase laboratory materials efficiently — a centralized "
            "digital marketplace for biological reagents, chemicals, and laboratory "
            "consumables, bridging suppliers and research institutions."
        ),
    },
    {
        "id": "project-sweetbloom",
        "text": (
            "Project: SweetBloom. An elegant e-commerce platform for SweetBloom, a luxury "
            "chocolate and flower boutique based in Qatar, featuring a product showcase, "
            "real-time inventory updates, secure order management, and a smooth checkout "
            "experience."
        ),
    },
    {
        "id": "project-qpet",
        "text": (
            "Project: QPet. A digital platform for QPet, a pet boarding service in Qatar, "
            "designed to provide personalized care, daily attention, and a calm environment "
            "for dogs and cats."
        ),
    },
    {
        "id": "testimonial-bmce",
        "text": (
            "Client testimonial: Dr. Mathew Thomas, Academic Director at BMCE, said Tresvance "
            "completely revitalized their college's web presence with an intuitive front-end "
            "for students and prospective applicants, and handled the transition flawlessly."
        ),
    },
    {
        "id": "testimonial-fintrust",
        "text": (
            "Client testimonial: Roy Samuel, Operations Head at FinTrust Banking, said "
            "Tresvance delivered a secure, scalable web platform tailored for their gold loan "
            "services, fitting their unique daily workflows with top-tier software engineering."
        ),
    },
    {
        "id": "testimonial-qpet",
        "text": (
            "Client testimonial: Dr. Sunil, Founder of QPET Clinic, said the digital "
            "transformation (web app and client data streamlining) exceeded expectations for "
            "both staff and pet owners."
        ),
    },
    {
        "id": "testimonial-sweetbloom",
        "text": (
            "Client testimonial: Anil Antony, Marketing Director at SWEETBLOOM, said the new "
            "e-commerce layout and targeted digital strategy doubled their online engagement "
            "within three months."
        ),
    },
    {
        "id": "testimonial-biohippo",
        "text": (
            "Client testimonial: Jibin Samuel, Lead Researcher at BIOHIPPO, said Tresvance's "
            "AI-integrated model saved countless hours of manual work in their research "
            "workflows."
        ),
    },
    {
        "id": "testimonial-westeast",
        "text": (
            "Client testimonial: Robin Sam, CTO at West-East, said Tresvance overhauled their "
            "legacy systems, implemented end-to-end cybersecurity, and provided proactive IT "
            "management and cloud infrastructure support."
        ),
    },
    {
        "id": "contact-info",
        "text": (
            "Contact Tresvance: Email info@tresvance.com or contact@tresvance.com. Phone: "
            "+91 8921187643. Tresvance is an IT Solutions & Managed Services company "
            "headquartered in Kochi, Kerala, India, available on-site in Kochi or virtually "
            "worldwide. To start a project, simply share your name, contact info, and "
            "project details right here in the chat — no need to visit a separate page."
        ),
    },
]

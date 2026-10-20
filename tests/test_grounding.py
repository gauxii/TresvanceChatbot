"""Issue 4 (unsupported industry/experience claims) and Issue 5 (negation
blind spot in the AI-attribution check). These test the deterministic
grounding functions directly with hand-built context, so they're fast and
don't depend on what the live LLM happens to say on a given run."""
import main

QPET_CONTEXT_IDS = [
    "testimonial-biohippo", "testimonial-qpet", "testimonial-bmce",
    "testimonial-sweetbloom", "testimonial-westeast", "contact-info",
]
QPET_CONTEXT_CHUNKS = [
    "Client testimonial: Jibin Samuel, Lead Researcher at BIOHIPPO, said Tresvance's "
    "AI-integrated model saved countless hours of manual work in their research workflows.",
    "Client testimonial: Dr. Sunil, Founder of QPET Clinic, said the digital "
    "transformation (web app and client data streamlining) exceeded expectations for "
    "both staff and pet owners.",
    "Project: QPet. A digital platform for QPet, a pet boarding service in Qatar, "
    "designed to provide personalized care, daily attention, and a calm environment "
    "for dogs and cats.",
    "Client testimonial: Anil Antony, Marketing Director at SWEETBLOOM, said the new "
    "e-commerce layout doubled their online engagement within three months.",
    "Client testimonial: Robin Sam, CTO at West-East, said Tresvance overhauled their "
    "legacy systems.",
    "Contact Tresvance: Email info@tresvance.com or contact@tresvance.com. "
    "Phone: +91 8921187643.",
]


# --- Issue 4: hospital/QPET regression and generalized industry claims ----

def test_hospital_claim_from_qpet_is_rejected():
    """The actual bug report: the model claimed healthcare/hospital
    experience, citing QPET (a pet boarding service) as evidence."""
    bad_answer = (
        "Yes, we've worked with healthcare institutions like QPET Clinic, where we "
        "helped them streamline their client data and improve the overall patient "
        "experience through a web app and digital transformation."
    )
    assert main.is_grounded(bad_answer, QPET_CONTEXT_IDS, QPET_CONTEXT_CHUNKS, "Have you worked with any hospitals?") is False


def test_unsupported_industry_terms_are_identified():
    bad_answer = (
        "Yes, we've worked with healthcare institutions, improving patient experience."
    )
    found = main.extract_unsupported_industry_claims(bad_answer, " ".join(QPET_CONTEXT_CHUNKS))
    assert "healthcare" in found
    assert "patient" in found


def test_honest_denial_of_hospital_experience_is_allowed():
    honest_answer = (
        "I don't have any record of working with hospitals or healthcare institutions. "
        "We have worked with QPET, a pet boarding service, on a digital platform."
    )
    assert main.is_grounded(honest_answer, QPET_CONTEXT_IDS, QPET_CONTEXT_CHUNKS, "Have you worked with any hospitals?") is True


def test_genuinely_supported_industry_claim_still_allowed():
    """Point 9 of Issue 4: don't block every answer that mentions an
    industry — a claim actually backed by context must pass."""
    ids = ["testimonial-fintrust"]
    chunks = [
        "Client testimonial: Roy Samuel, Operations Head at FinTrust Banking, said "
        "Tresvance delivered a secure, scalable web platform tailored for their gold "
        "loan services."
    ]
    true_claim = "Yes, we have experience in banking — we built a platform for FinTrust Banking."
    assert main.is_grounded(true_claim, ids, chunks, "Have you worked with banks?") is True


def test_pet_care_claim_about_qpet_itself_is_fine():
    """Describing QPET accurately (pet boarding, not healthcare) must not be
    flagged — only the false healthcare/hospital framing should be."""
    accurate_answer = (
        "We built a digital platform for QPET, a pet boarding service in Qatar, "
        "providing personalized care for dogs and cats."
    )
    assert main.is_grounded(accurate_answer, QPET_CONTEXT_IDS, QPET_CONTEXT_CHUNKS, "Tell me about QPet.") is True


def test_other_unsupported_industry_examples():
    """Regression coverage beyond the single hospital example, per Issue 4
    point 10 ('and other unsupported industry-experience claims')."""
    cases = [
        "We've delivered several projects for government agencies.",
        "We have deep experience serving pharmaceutical manufacturers.",
        "Our team has built platforms for major airlines and the hospitality industry.",
    ]
    for answer in cases:
        assert main.is_grounded(answer, QPET_CONTEXT_IDS, QPET_CONTEXT_CHUNKS, "What industries have you worked in?") is False, answer


# --- Issue 5: negation blind spot in AI-attribution check ------------------

SWEETBLOOM_IDS = ["project-sweetbloom", "testimonial-sweetbloom"]
SWEETBLOOM_CHUNKS = [
    "Project: SweetBloom. An elegant e-commerce platform for SweetBloom, a luxury "
    "chocolate and flower boutique based in Qatar, featuring a product showcase, "
    "real-time inventory updates, secure order management, and a smooth checkout "
    "experience.",
    "Client testimonial: Anil Antony, Marketing Director at SWEETBLOOM, said the new "
    "e-commerce layout and targeted digital strategy doubled their online engagement "
    "within three months.",
]


def test_correct_negative_ai_claim_is_not_rejected():
    """The actual bug report: a truthful 'No, it wasn't AI-powered' answer
    was wrongly rejected and replaced with an unhelpful fallback."""
    negative_answer = (
        "No, the SweetBloom project was not an AI project. It was an e-commerce "
        "platform focused on product showcase, inventory, and checkout."
    )
    assert main.is_grounded(negative_answer, SWEETBLOOM_IDS, SWEETBLOOM_CHUNKS, "Was the SweetBloom project AI-powered?") is True


def test_false_positive_ai_claim_is_still_rejected():
    """The negation fix must not become a blanket pass — an actual false
    positive claim in the same shape has to keep failing."""
    false_claim = (
        "Yes, SweetBloom was an AI-powered project using machine learning to "
        "personalize recommendations."
    )
    assert main.is_grounded(false_claim, SWEETBLOOM_IDS, SWEETBLOOM_CHUNKS, "Was the SweetBloom project AI-powered?") is False


def test_uncertain_ai_claim_phrasing_is_treated_as_a_denial():
    uncertain_answer = "I'm not sure SweetBloom used AI — the context doesn't mention it."
    assert main.is_grounded(uncertain_answer, SWEETBLOOM_IDS, SWEETBLOOM_CHUNKS, "Was the SweetBloom project AI-powered?") is True


def test_negation_elsewhere_in_answer_does_not_mask_unrelated_false_claim():
    """Issue 5 point 5: a simplistic fix must not treat every 'not'/'never'
    anywhere in the answer as proof the whole answer is safe — negation must
    stay scoped to the sentence making the claim."""
    mixed_answer = (
        "We have never discussed pricing publicly. SweetBloom was an AI-powered "
        "project using machine learning."
    )
    assert main.is_grounded(mixed_answer, SWEETBLOOM_IDS, SWEETBLOOM_CHUNKS, "Was the SweetBloom project AI-powered?") is False


def test_supported_positive_ai_claim_still_allowed():
    ids = ["testimonial-biohippo"]
    chunks = [
        "Client testimonial: Jibin Samuel, Lead Researcher at BIOHIPPO, said "
        "Tresvance's AI-integrated model saved countless hours of manual work in "
        "their research workflows."
    ]
    answer = "Yes, the BioHippo project used an AI-integrated model."
    assert main.is_grounded(answer, ids, chunks, "Was BioHippo an AI project?") is True

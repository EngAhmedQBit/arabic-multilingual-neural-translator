# ============================================================
# test_translator.py
# Command-line test for the Arabic-to-Multiple-Languages translator
# ============================================================

from translator_core import ArabicTranslator, SUPPORTED_LANGUAGES


def test_single_translation():
    """Test translating a single sentence to all supported languages."""
    translator = ArabicTranslator()

    test_text = "مرحباً، أنا طالب في جامعة برلين وأدرس معالجة اللغات الطبيعية."

    print("=" * 70)
    print("🔹 Arabic-to-Multiple-Languages Translation Test")
    print("=" * 70)
    print(f"🇸🇦 AR: {test_text}")
    print("-" * 70)

    for code, info in SUPPORTED_LANGUAGES.items():
        try:
            result = translator.translate(test_text, target_lang=code)
            print(f"{info['name']:<15} {result}")
        except Exception as e:
            print(f"{info['name']:<15} ⚠️ Error: {e}")

    print("=" * 70)


def test_batch_translation():
    """Test batch translation to one language."""
    translator = ArabicTranslator()

    test_dataset = [
        "السلام عليكم",
        "كيف حالك اليوم؟",
        "أنا أحب البرمجة كثيراً.",
        "الطقس جميل في فصل الربيع.",
        "أريد أن أسافر إلى ألمانيا العام المقبل.",
        "هذا الكتاب مفيد جداً للمبتدئين.",
        "التعلم الآلي فرع مهم من الذكاء الاصطناعي.",
        "شكراً على مساعدتك.",
    ]

    print("\n" + "=" * 70)
    print("🔹 Batch Translation Test (Arabic → English)")
    print("=" * 70)
    print(f"{'#':<3} {'الجملة العربية':<40} {'الترجمة الإنجليزية'}")
    print("-" * 70)

    results = translator.translate_batch(test_dataset, target_lang="en")

    for i, (ar, en) in enumerate(zip(test_dataset, results), 1):
        print(f"{i:<3} {ar:<40} {en}")

    print("=" * 70)
    print("✅ All tests completed successfully!")


if __name__ == "__main__":
    test_single_translation()
    test_batch_translation()

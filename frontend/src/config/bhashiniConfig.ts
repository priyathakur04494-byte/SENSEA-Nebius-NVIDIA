/**
 * BHASHINI API CREDENTIALS
 * -------------------------
 * Bhashini (https://bhashini.gov.in) is India's government-run multilingual
 * AI API — free for approved use and purpose-built for Indian language
 * translation, which is why it's the default provider here.
 *
 * HOW TO GET YOUR OWN CREDENTIALS (required — the app will not translate
 * without these):
 *   1. Go to https://bhashini.gov.in and sign up for API access, or go
 *      through https://bhashini.gitbook.io/bhashini-apis for the current
 *      onboarding flow (Bhashini's docs/portal do change over time, so
 *      follow whatever their site currently shows).
 *   2. Once approved, you'll receive a `userID` and a `ulcaApiKey`.
 *   3. Paste them below. Do NOT commit real keys to a public repo — for a
 *      real app, load these from an environment variable or a secure
 *      secrets store instead of hardcoding them.
 */

export const BHASHINI_CONFIG = {
  userID: "YOUR_BHASHINI_USER_ID",
  ulcaApiKey: "YOUR_BHASHINI_ULCA_API_KEY",
  // These endpoints are current as of Bhashini's published docs but the
  // service has changed URLs before — verify against their docs if calls
  // start failing.
  pipelineEndpoint: "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline",
  pipelineId: "64392f96daac500b55c543cd", // Bhashini's default translation pipeline
};

export function isBhashiniConfigured(): boolean {
  return (
    BHASHINI_CONFIG.userID !== "YOUR_BHASHINI_USER_ID" &&
    BHASHINI_CONFIG.ulcaApiKey !== "YOUR_BHASHINI_ULCA_API_KEY" &&
    BHASHINI_CONFIG.userID.length > 0 &&
    BHASHINI_CONFIG.ulcaApiKey.length > 0
  );
}

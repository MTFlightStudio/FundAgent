# Testing Enhanced Market Research

This guide shows you how to test the improved market research agent that uses HubSpot data for specific market targeting.

## 🧪 Testing Methods

### 1. Quick Unit Test ✅ (Already Working)

Test the market definition logic:

```bash
python test_market_definition.py
```

**Expected Output**: Should show specific markets like "Public Transit Technology Market" instead of generic "Transportation"

---

### 2. CLI Testing (Compare Old vs New)

#### A) Test OLD approach (generic):

```bash
python -m ai_agents.agents.market_intelligence_agent --sector "Transportation"
```

This will research the broad Transportation industry.

#### B) Test NEW approach (specific with HubSpot data):

```bash
python -m ai_agents.agents.market_intelligence_agent \
  --sector "Transportation" \
  --hubspot-data test_hubspot_sample.json
```

This should research "Public Transit Technology Market" specifically.

---

### 3. Test with Different Market Types

Create different HubSpot test files to see various market targeting:

#### Kefir Company Test:

```bash
# Create kefir_test.json
cat > kefir_test.json << EOF
{
  "associated_companies": [{
    "properties": {
      "name": "ProBio Kefir Co.",
      "describe_the_business_product_in_one_sentence": "We produce organic kefir and probiotic beverages for health-conscious consumers.",
      "what_sector_is_your_business_product_": "Health and Wellness",
      "where_is_your_business_based_": "London, UK"
    }
  }]
}
EOF

# Test it
python -m ai_agents.agents.market_intelligence_agent \
  --sector "Health and Wellness" \
  --hubspot-data kefir_test.json
```

**Expected**: Should research "Kefir and Probiotic Beverages Market" instead of broad "Health and Wellness"

#### BNPL FinTech Test:

```bash
# Create bnpl_test.json
cat > bnpl_test.json << EOF
{
  "associated_companies": [{
    "properties": {
      "name": "PayLater Pro",
      "describe_the_business_product_in_one_sentence": "We provide buy now pay later solutions for online retailers.",
      "what_sector_is_your_business_product_": "FinTech",
      "where_is_your_business_based_": "San Francisco, CA, USA"
    }
  }]
}
EOF

# Test it
python -m ai_agents.agents.market_intelligence_agent \
  --sector "FinTech" \
  --hubspot-data bnpl_test.json
```

**Expected**: Should research "Buy Now Pay Later (BNPL) Market" instead of broad "FinTech"

---

### 4. End-to-End Testing with Real Deal

If you have a test deal ID in your HubSpot:

```bash
python -m ai_agents.agents.investment_orchestrator
```

This will run the full orchestrator with a test deal and automatically use enhanced market research.

---

### 5. Verify Output Quality

Look for these improvements in the market research output:

#### ✅ **Specific Market Sizing**

- OLD: "Transportation market size: $7T globally"
- NEW: "Public Transit Technology market size: $45B globally"

#### ✅ **Targeted Competitors**

- OLD: General transportation companies (UPS, FedEx, etc.)
- NEW: Transit tech companies (Moovit, Citymapper, Transit app, etc.)

#### ✅ **Relevant Trends**

- OLD: Broad transportation trends (e-commerce, logistics)
- NEW: Transit-specific trends (digital payments, real-time tracking, MaaS)

#### ✅ **Geographic Focus**

- OLD: Global transportation market
- NEW: Caribbean/Barbados transit market with regional context

#### ✅ **Precise Barriers**

- OLD: Generic transportation barriers
- NEW: Transit app specific barriers (regulatory approval, data partnerships, etc.)

---

## 🔍 What to Look For

### Success Indicators:

1. **Market Focus**: JSON should show specific market in `industry_overview`
2. **Geographic Scope**: `jurisdiction` should match company location
3. **Targeted Competitors**: `competitors` should be niche players, not broad industry
4. **Specific Market Size**: `market_size_tam` should be segment-specific
5. **Relevant Trends**: `key_market_trends` should be product/service specific

### Log Messages to Watch:

```
✅ GOOD: "Using specific market definition - Public Transit Technology Market"
✅ GOOD: "Enhanced analysis completed for specific market"
❌ OLD:  "Using generic market research for: Transportation"
```

---

## 🚨 Troubleshooting

### If you see generic results instead of specific:

1. **Check HubSpot Data**: Ensure the JSON has `describe_the_business_product_in_one_sentence`
2. **Verify Keywords**: The business description should contain recognizable keywords
3. **Check Logs**: Look for "market definition" messages in the output
4. **Test Pattern Matching**: Use `test_market_definition.py` to verify your data triggers patterns

### If the market research fails:

1. **API Keys**: Ensure `TAVILY_API_KEY` is set for search functionality
2. **Dependencies**: Check all imports work with `python -c "from ai_agents.agents.market_intelligence_agent import *"`
3. **Fallback**: The system should fall back to generic research if HubSpot data fails

---

## 🎯 Quick Test Commands

Run these in sequence to test everything:

```bash
# 1. Unit test
python test_market_definition.py

# 2. CLI comparison test
python -m ai_agents.agents.market_intelligence_agent --sector "Transportation"
python -m ai_agents.agents.market_intelligence_agent --sector "Transportation" --hubspot-data test_hubspot_sample.json

# 3. Full demonstration
python test_enhanced_market_research.py
```

If all three work, your enhanced market research is ready! 🚀

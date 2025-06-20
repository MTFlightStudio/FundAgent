from ai_agents.agents.market_intelligence_agent import run_market_intelligence_cli
import json

# Test with sample HubSpot data that should trigger enhanced targeting
hubspot_data = {
    'deal': {'id': '241939260658'},
    'associated_companies': [{
        'properties': {
            'name': 'Shopthru',
            'what_sector_is_your_business_product_': 'Financial Services/ Fintech',
            'describe_the_business_product_in_one_sentence': 'We provide payment processing and checkout solutions for e-commerce businesses',
            'what_is_your_usp__what_makes_you_different_from_your_competitors_': 'Seamless one-click payment experience with advanced fraud detection'
        }
    }]
}

print('🧪 Testing enhanced market research with Shopthru data...')
result = run_market_intelligence_cli('Financial Services/ Fintech', hubspot_data=hubspot_data)
print('✅ Enhanced market research completed')

if result:
    if hasattr(result, 'target_market_segment'):
        print(f'🎯 Target market: {result.target_market_segment}')
    if hasattr(result, 'market_size_tam'):
        print(f'💰 Market size: {result.market_size_tam}')
    print(f'📊 Result type: {type(result).__name__}')
else:
    print('❌ No result returned') 
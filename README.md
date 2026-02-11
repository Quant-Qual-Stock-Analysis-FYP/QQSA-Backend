# Setup Instructions

## 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## 2. Install Dependencies
```bash
pip install -r requirements.txt
# Upgrade to the latest version of sentence-transformers (recommended)
pip install -U sentence-transformers
```
> The above command will install the latest version of sentence-transformers.

## 3. Environment Variables
Create a `.env` file in the backend directory with:
```
DEEPSEEK_API_KEY=your_deepseek_api_key
FINNHUB_API_KEY=your_finnhub_api_key
```

## 4. Database Migrations
```bash
# Create migrations
python manage.py makemigrations

# Apply migrations
python manage.py migrate
```

## 5. Run Server
```bash
python manage.py runserver
```

## Deactivate Virtual Environment
```bash
deactivate
```

## 注意事项

1. **缓存依赖**: 确保Django缓存后端已配置（Redis或Memcached）
2. **日志配置**: 确保Django日志配置正确，以便查看错误信息
3. **向后兼容**: 所有改进都保持向后兼容，不影响现有功能
4. **性能监控**: 建议监控缓存命中率和数据库查询性能

## 后续建议

1. **单元测试**: 为新增的验证和缓存功能添加单元测试
2. **性能监控**: 添加性能监控指标（查询时间、缓存命中率等）
3. **配置热更新**: 考虑支持配置的动态更新
4. **缓存失效策略**: 优化缓存失效策略，提高缓存命中率


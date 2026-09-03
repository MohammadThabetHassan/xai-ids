# Security Policy

## Supported Versions

| Version | Supported                        |
|---------|----------------------------------|
| 3.0.x   | Yes                              |
| 2.0.x   | Security fixes only              |
| < 2.0   | No                               |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly:

1. **Do NOT** open a public issue
2. Contact the maintainers directly via GitHub
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Known Security Considerations

- **Model Deserialization**: The API uses `joblib.load()` to load models. Only load models from trusted sources.
- **API Authentication**: The FastAPI endpoint has no authentication. Do not expose it publicly without adding auth.
- **Input Validation**: The API validates input length but does not validate feature ranges or distributions.
- **No Rate Limiting**: The API has no rate limiting. Consider adding one for production use.

## Best Practices for Deployment

1. Never expose the API directly to the internet without authentication
2. Use HTTPS in production
3. Validate and sanitize all inputs
4. Keep dependencies updated
5. Use Docker for isolation
6. Monitor API logs for suspicious activity

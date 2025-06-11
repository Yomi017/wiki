---
{"dg-publish":true,"permalink":"/notion/practical-knowledge/computer-science/development/web/"}
---

# 1. JWT 令牌：

**前端发送的形式有：**

1. Bearer TOKEN_STRING
2. TOKEN_STRING
3. JWT TOKEN_STRING

注意：TOKEN_STRING 是没有引号的

  

**后端发送id等数字时需转为string：**

```Python
@bp.route('/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    
    if not data.get('username') or not data.get('password'):
        return jsonify({"msg": "Username and password required"}), 400
    
    user = User.query.filter_by(username=data.get('username'), is_deleted=False).first()
    if user is None or not user.check_password(data.get('password')):
        return jsonify({"msg": "Invalid username or password"}), 401

    # Create both access and refresh tokens
    # 注意：这里数字记得转换成string
    access_token = create_access_token(identity=str(user.id))
    refresh_token = create_refresh_token(identity=str(user.id))
    
    # Return both tokens and user information
    return jsonify({
        "access_token": access_token,
        "refresh_token": refresh_token,
        "user": user.to_dict(include_contact=True)
    }), 200
```
from django.contrib.auth.models import AbstractUser
from django.db import models

class CustomUser(AbstractUser):
    # 這裡可以加自定義欄位，例如手機號碼等
    # Django 預設已有 username, email, password, first_name, last_name
    email = models.EmailField(unique=True) # 強制 email 唯一

    def __str__(self):
        return self.username
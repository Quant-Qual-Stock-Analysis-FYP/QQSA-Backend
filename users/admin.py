from django.contrib import admin
from django.contrib.auth.admin import UserAdmin # 1. 引入專用的 UserAdmin
from .models import CustomUser

# 2. 繼承 UserAdmin 而不是 ModelAdmin
@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    # 設定列表顯示的欄位 (建議加上 is_staff 方便辨識管理員)
    list_display = ('id', 'username', 'email')
    
    # 設定搜尋欄位
    search_fields = ('username', 'email')

    # UserAdmin 已經幫你寫好了密碼加密邏輯、修改密碼表單等功能
    # 所以不需要像普通 Model 那樣自己處理
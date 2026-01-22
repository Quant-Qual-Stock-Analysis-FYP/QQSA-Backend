# Generated migration to create default chatrooms for all users

from django.conf import settings
from django.db import migrations


def create_default_chatrooms_for_all_users(apps, schema_editor):
    """Create a default chatroom for all users who don't have one"""
    ChatRoom = apps.get_model('chatbot', 'ChatRoom')
    User = apps.get_model(settings.AUTH_USER_MODEL)
    
    # Get all users
    all_users = User.objects.all()
    
    # Create a default chat room for each user who doesn't have one
    for user in all_users:
        if not ChatRoom.objects.filter(user=user).exists():
            ChatRoom.objects.create(
                user=user,
                title="Default Chat"
            )


def reverse_migration(apps, schema_editor):
    """Reverse migration: delete default chatrooms (optional, can be no-op)"""
    # We don't need to reverse this - if users want to delete their default chatroom, they can
    pass


class Migration(migrations.Migration):

    dependencies = [
        ('chatbot', '0005_alter_chatmessage_options'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.RunPython(
            create_default_chatrooms_for_all_users,
            reverse_migration
        ),
    ]

# Generated migration for ChatRoom feature

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


def migrate_existing_messages_to_chatrooms(apps, schema_editor):
    """Migrate existing ChatMessages to ChatRooms"""
    ChatRoom = apps.get_model('chatbot', 'ChatRoom')
    ChatMessage = apps.get_model('chatbot', 'ChatMessage')
    User = apps.get_model(settings.AUTH_USER_MODEL)
    
    # Get all users who have messages
    users_with_messages = User.objects.filter(
        chat_history__isnull=False
    ).distinct()
    
    # Create a default chat room for each user and migrate their messages
    for user in users_with_messages:
        # Create a default chat room for this user
        chat_room = ChatRoom.objects.create(
            user=user,
            title=f"Default Chat"  # Default title
        )
        
        # Migrate all messages from this user to the new chat room
        ChatMessage.objects.filter(user=user).update(chat_room=chat_room)


def reverse_migration(apps, schema_editor):
    """Reverse migration: assign messages back to users"""
    # Note: This function runs after the user field is added back to ChatMessage
    ChatRoom = apps.get_model('chatbot', 'ChatRoom')
    ChatMessage = apps.get_model('chatbot', 'ChatMessage')
    
    # For each chat room, assign messages back to the room's user
    for chat_room in ChatRoom.objects.all():
        ChatMessage.objects.filter(chat_room=chat_room).update(user=chat_room.user)


class Migration(migrations.Migration):

    dependencies = [
        ('chatbot', '0003_recommendedquestion'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        # Step 1: Create ChatRoom model
        migrations.CreateModel(
            name='ChatRoom',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(blank=True, help_text='聊天室标题，可选', max_length=200)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='chat_rooms', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': '聊天室',
                'verbose_name_plural': '聊天室',
                'ordering': ['-updated_at'],
            },
        ),
        
        # Step 2: Add chat_room field as nullable first
        migrations.AddField(
            model_name='chatmessage',
            name='chat_room',
            field=models.ForeignKey(
                null=True,
                blank=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name='messages',
                to='chatbot.chatroom'
            ),
        ),
        
        # Step 3: Migrate existing data
        migrations.RunPython(
            migrate_existing_messages_to_chatrooms,
            reverse_migration
        ),
        
        # Step 4: Make chat_room non-nullable
        migrations.AlterField(
            model_name='chatmessage',
            name='chat_room',
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name='messages',
                to='chatbot.chatroom'
            ),
        ),
        
        # Step 5: Remove the old user field
        migrations.RemoveField(
            model_name='chatmessage',
            name='user',
        ),
    ]

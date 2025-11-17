VERSION=$(uv run python - << 'EOF'
import tomllib
print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])
EOF
)

git tag "v$VERSION"
git push origin "v$VERSION"

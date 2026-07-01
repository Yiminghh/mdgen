REMOTE ?= hephaestus
REMOTE_DIR ?= /home/yh1924/projects/mdgen

RSYNC_EXCLUDES = \
	--exclude='.git' \
	--exclude='__pycache__' \
	--exclude='.DS_Store' \
	--exclude='.idea' \
	--exclude='.venv/' \
	--exclude='venv/' \
	--exclude='envs/' \
	--exclude='data/' \
	--exclude='datasets/' \
	--exclude='outputs/' \
	--exclude='results/' \
	--exclude='checkpoints/' \
	--exclude='wandb/' \
	--exclude='logs/' \
	--exclude='tmp/'

.PHONY: print-sync-config sync-up _assert-remote-dir

print-sync-config:
	@echo "REMOTE=$(REMOTE)"
	@echo "REMOTE_DIR=$(REMOTE_DIR)"

_assert-remote-dir:
	@test -n "$(REMOTE)" || { echo "[make] ERROR: REMOTE is empty" >&2; exit 1; }
	@test -n "$(REMOTE_DIR)" || { echo "[make] ERROR: REMOTE_DIR is empty" >&2; exit 1; }
	@test "$(REMOTE_DIR)" != "/" || { echo "[make] ERROR: refusing to sync to /" >&2; exit 1; }
	@test "$(REMOTE_DIR)" != "." || { echo "[make] ERROR: refusing to sync to ." >&2; exit 1; }

sync-up: _assert-remote-dir
	@echo "[sync-up] remote=$(REMOTE) dir=$(REMOTE_DIR)"
	ssh $(REMOTE) "mkdir -p '$(REMOTE_DIR)'"
	rsync -avz --delete $(RSYNC_EXCLUDES) ./ $(REMOTE):$(REMOTE_DIR)/

provider "nebius" {
  service_account = {
    private_key_file_env = "AUTHKEY_PRIVATE_PATH"
    public_key_id_env    = "AUTHKEY_PUBLIC_ID"
    account_id_env       = "SA_ID"
  }
}

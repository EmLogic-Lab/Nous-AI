document.addEventListener("DOMContentLoaded", () => {

    const body = document.body;

    const collapseBtn =
        document.getElementById("collapseBtn");

    const collapsedBrand =
        document.getElementById("collapsedBrand");

    const mobileMenuBtn =
        document.getElementById("mobileMenuBtn");

    const newChatBtn =
        document.getElementById("newChatBtn");

    const sidebarOverlay =
        document.getElementById("sidebarOverlay");

    const openRecentFromSidebar =
        document.getElementById("openRecentFromSidebar");

    const recentPopup =
        document.getElementById("recentPopup");

    const recentPopupClose =
        document.getElementById("recentPopupClose");

    const textarea =
        document.getElementById("chatInput");

    const chatArea =
        document.getElementById("chatArea");

    const chatBottom =
        document.getElementById("chatBottom");

    const welcomeSection =
        document.getElementById("welcomeSection");

    const sendBtn =
        document.getElementById("sendBtn");


    /* =========================================
       SIDEBAR
    ========================================== */

    collapseBtn.addEventListener("click", () => {

        if (window.innerWidth <= 640) {

            body.classList.remove(
                "mobile-sidebar-open"
            );

            return;
        }


        body.classList.add(
            "sidebar-collapsed"
        );

        closeRecent();
    });


    collapsedBrand.addEventListener("click", () => {

        body.classList.remove(
            "sidebar-collapsed"
        );

        closeRecent();
    });


    /* =========================================
       MOBILE SIDEBAR
    ========================================== */

    function openMobileSidebar() {

        body.classList.add(
            "mobile-sidebar-open"
        );
    }


    function closeMobileSidebar() {

        body.classList.remove(
            "mobile-sidebar-open"
        );
    }


    mobileMenuBtn.addEventListener(
        "click",
        openMobileSidebar
    );


    sidebarOverlay.addEventListener(
        "click",
        closeMobileSidebar
    );


    /* =========================================
       RECENT OVERLAY
    ========================================== */

    function openRecent() {

        const recentButton =
            document.getElementById(
                "openRecentFromSidebar"
            );

        const buttonRect =
            recentButton.getBoundingClientRect();

        recentPopup.style.top =
            `${buttonRect.top}px`;

        recentPopup.style.left =
            `${buttonRect.right + 16}px`;

        recentPopup.classList.add(
            "open"
        );

        recentPopup.setAttribute(
            "aria-hidden",
            "false"
        );
    }


    function closeRecent() {

        recentPopup.classList.remove(
            "open"
        );

        recentPopup.setAttribute(
            "aria-hidden",
            "true"
        );
    }


    /*
     * Recent button inside the sidebar.
     *
     * When collapsed:
     * Opens the overlay.
     *
     * When open:
     * Does nothing because recent chats
     * are already visible in the sidebar.
     */

    openRecentFromSidebar.addEventListener(
        "click",
        (event) => {

            event.stopPropagation();


            if (
                body.classList.contains(
                    "sidebar-collapsed"
                )
            ) {

                if (
                    recentPopup.classList.contains(
                        "open"
                    )
                ) {

                    closeRecent();

                } else {

                    openRecent();

                }

            }

        }
    );


    recentPopupClose.addEventListener(
        "click",
        closeRecent
    );


    /*
     * Clicking outside Recent closes it.
     */

    document.addEventListener(
        "click",
        (event) => {

            if (
                recentPopup.classList.contains(
                    "open"
                ) &&
                !recentPopup.contains(
                    event.target
                ) &&
                !openRecentFromSidebar.contains(
                    event.target
                )
            ) {

                closeRecent();

            }

        }
    );


    /* =========================================
       NEW CHAT
    ========================================== */

    function startNewChat() {

        textarea.value = "";

        textarea.style.height =
            "auto";

        welcomeSection.hidden =
            false;

        chatArea.scrollTop =
            0;

        closeRecent();


        if (window.innerWidth <= 640) {

            closeMobileSidebar();

        }


        textarea.focus();
    }


    newChatBtn.addEventListener(
        "click",
        startNewChat
    );


    /* =========================================
       TEXTAREA
    ========================================== */

    function resizeTextarea() {

        textarea.style.height =
            "auto";


        const maxHeight =
            window.innerWidth <= 640
                ? 150
                : 200;


        textarea.style.height =
            `${Math.min(
                textarea.scrollHeight,
                maxHeight
            )}px`;
    }


    textarea.addEventListener(
        "input",
        resizeTextarea
    );


    /* =========================================
       SEND
    ========================================== */

    async function sendMessage() {

        const message =
            textarea.value.trim();


        if (!message) {
            return;
        }


        welcomeSection.hidden = true;


        // ================================
        // USER MESSAGE
        // ================================

        const userMessage =
            document.createElement("div");

        userMessage.className =
            "user-message";

        userMessage.textContent =
            message;

        chatArea.appendChild(
            userMessage
        );


        textarea.value = "";

        textarea.style.height = "auto";

        chatArea.scrollTop =
            chatArea.scrollHeight;


        // ================================
        // ASK NOUS BACKEND
        // ================================

        try {

            const response =
                await fetch("https://nous-ai-backend-f4lc.onrender.com/api/chat", {

                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        message: message
                    })

                });


            if (!response.ok) {

                throw new Error(
                    `Server returned ${response.status}`
                );

            }


            const data =
                await response.json();


            // ================================
            // NOUS RESPONSE
            // ================================

            const assistantMessage =
                document.createElement("div");

            assistantMessage.className =
                "assistant-message";

            assistantMessage.textContent =
                data.reply;

            chatArea.appendChild(
                assistantMessage
            );


            chatArea.scrollTop =
                chatArea.scrollHeight;


        } catch (error) {

            console.error(
                "Nous AI request failed:",
                error
            );


            const errorMessage =
                document.createElement("div");

            errorMessage.className =
                "assistant-message";   

            errorMessage.textContent =
                "Sorry, Nous could not connect to the backend.";

            chatArea.appendChild(
                errorMessage
            );


            chatArea.scrollTop =
                chatArea.scrollHeight;

        }
    }


    sendBtn.addEventListener(
        "click",
        sendMessage
    );


    /* =========================================
       DESKTOP ENTER
    ========================================== */

    textarea.addEventListener(
        "keydown",
        (event) => {

            if (
                event.key === "Enter" &&
                !event.shiftKey &&
                window.innerWidth > 640
            ) {

                event.preventDefault();

                sendMessage();

            }

        }
    );


    /* =========================================
       MOBILE KEYBOARD
    ========================================== */

    if (window.visualViewport) {

        let lastKeyboardHeight = 0;


        window.visualViewport.addEventListener(
            "resize",
            () => {

                if (window.innerWidth > 640) {
                    return;
                }


                const keyboardHeight =
                    Math.max(
                        0,
                        window.innerHeight -
                        window.visualViewport.height
                    );


                if (
                    Math.abs(
                        keyboardHeight -
                        lastKeyboardHeight
                    ) < 8
                ) {

                    return;

                }


                lastKeyboardHeight =
                    keyboardHeight;


                if (keyboardHeight > 100) {

                    chatBottom.style.transform =
                        `translate3d(
                            0,
                            -${keyboardHeight}px,
                            0
                        )`;

                } else {

                    chatBottom.style.transform =
                        "translate3d(0, 0, 0)";

                }

            },
            {
                passive: true
            }
        );
    }


    /* =========================================
       ESCAPE
    ========================================== */

    document.addEventListener(
        "keydown",
        (event) => {

            if (
                event.key !== "Escape"
            ) {

                return;

            }


            closeRecent();


            if (
                window.innerWidth <= 640
            ) {

                closeMobileSidebar();

            }

        }
    );


    /* =========================================
       RESIZE
    ========================================== */

    window.addEventListener(
        "resize",
        () => {
            if (
                recentPopup.classList.contains("open") &&
                window.innerWidth > 640
            ) {
                const recentButton =
                    document.getElementById(
                        "openRecentFromSidebar"
                    );

                const buttonRect =
                    recentButton.getBoundingClientRect();

                recentPopup.style.top =
                    `${buttonRect.top}px`;

                recentPopup.style.left =
                    `${buttonRect.right + 10}px`;
            }

            if (
                window.innerWidth > 640
            ) {

                closeMobileSidebar();

                chatBottom.style.transform =
                    "translate3d(0, 0, 0)";

            }

        },
        {
            passive: true
        }
    );

});